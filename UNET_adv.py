
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import numpy as np
import tensorflow.keras.optimizers as optimizer
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Reshape, Activation, concatenate, Dropout
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History, ReduceLROnPlateau, LearningRateScheduler
import tensorflow as tf
import os
import sys

if int(tf.__version__[0])<2:
    tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
    print('eager execution is activated. Tensorflow version is: ' +str(tf.__version__[0]))

wd=os.getcwd()

class UNET(): #Main class. All methodes deemed usefull to achief more robust image segmentation

   
    def __init__(self,conv_min_depth=32,conv_max_depth=512,data=None,mask=None,labels=[]):# Constructor. Generates dimensions of the network depending on varaibles 
        self.data=data
        self.mask=mask
        self.label=labels
        self.conv_depth=conv_max_depth
        self.step_size=[conv_min_depth,conv_max_depth]
        self.transfer=[]
 
        
    def patterns(self,c_size=5120,boundry=256,images=1000,draw=True):#pattern generator, Produces random patterns of variable size comprized of triangles squares circles and stars.
        from numpy.random import randint as random
        from skimage.draw import line
        xy=[0,np.pi/6,np.pi/4,np.pi/3,np.pi/2,2*np.pi/3,3*np.pi/4,5*np.pi/6,np.pi,7*np.pi/6,5*np.pi/4,4*np.pi/3,3*np.pi/2,5*np.pi/3,7*np.pi/4,11*np.pi/6]
        up=np.sin(xy)*20
        up=np.where(up-np.floor(up)>.5,np.ceil(up),np.floor(up))
        side=np.cos(xy)*20
        side=np.where(side-np.floor(side)>.5,np.ceil(side),np.floor(side))
        classes=4
        canvas=np.zeros(shape=(c_size,c_size,classes),dtype=np.float32)
        star_canvas=np.zeros(shape=(c_size,c_size),dtype=np.float32)
        square_canvas=np.zeros(shape=(c_size,c_size),dtype=np.float32)
        circle_canvas=np.zeros(shape=(c_size,c_size),dtype=np.float32)
        triangle_canvas=np.zeros(shape=(c_size,c_size),dtype=np.float32)
        star=[1,7,14,4,10]
        square=[i for i in range(0,len(up),4)]
        triangle=[0,6,11]
        circle=[i for i in range(len(up))]
        forms=[triangle,star,square,circle]
        canvases=[triangle_canvas,star_canvas,square_canvas,circle_canvas]
        maybe='x'
        for example in range(int((c_size**2/boundry**2)*1.5)):
            sys.stdout.write('|'+' \r' if example%2==0 else '-'+' \r')
            sys.stdout.flush()
            tryoi=np.zeros((boundry,boundry))
            size=random(1,7)
            posx=int(boundry/2)#np.random.randint(size+np.max(up)+1,boundry-size-np.max(up)-1)
            posy=int(boundry/2)#np.random.randint(size+np.max(side)+1,boundry-size-np.max(side)-1)
            tilt=random(0,len(up))
            types=random(0,classes)
            points=np.array([],dtype=np.int32)
            for i in forms[types]:
                i=(i+tilt)%len(up)
                x=int(size*up[i]+posx)%boundry
                y=int(size*side[i]+posy)%boundry
                tryoi[x,y]=1
                pos=np.array([x,y])
                points=np.append(points,pos)
            points=points.reshape((len(forms[types]),2))
            for pos in range(len(points)):
                x,y=points[pos-1]
                xx,yy=points[pos]
                rr,cc=line(x,y,xx,yy)
                tryoi[rr,cc]=1

            for i in range(boundry):
                for ii in range(boundry-1):
                    select=0
                    if tryoi[i,ii]==1 and tryoi[i,ii+1]==0:
                        for u in range(boundry-i):
                            if tryoi[i+u,ii+1]==1:
                                select+=1
                                break
                        for u in range(i):
                            if tryoi[i-u,ii+1]==1:
                                select+=1
                                break
                        for u in range(boundry-ii-1):
                            if tryoi[i,ii+u+1]==1:
                                select+=1
                                break 
                        if select==3:
                            tryoi[i,ii+1]=1
            may=0
            while True:
                pos_c=random(c_size-boundry)
                pos_v=random(c_size-boundry)
                color=random(4)
                test=np.zeros(shape=(boundry,boundry,4))
                test+=canvas[pos_c:pos_c+boundry,pos_v:pos_v+boundry]
                test[:,:,color]+=tryoi
                if np.mean(canvas[pos_c:pos_c+boundry,pos_v:pos_v+boundry])<=may and np.max(np.sum(test,axis=-1))<=1:
                    canvas[pos_c:pos_c+boundry,pos_v:pos_v+boundry,color]+=tryoi
                    canvases[types][pos_c:pos_c+boundry,pos_v:pos_v+boundry]+=tryoi
                    break
                may+=1/boundry**2
                if may>12/boundry:
                    maybe='o' if maybe=='x' else 'x'
                    sys.stdout.write(maybe+' \r')
                    sys.stdout.flush()
                    break
                    
        for x in [0,2]:
            canvas[:,:,x]+=canvas[:,:,3]
        canvas=np.clip(canvas[:,:,:3],0,1)
        canvases.append(np.where(np.sum(canvas,axis=2)==0,1,0))
        canvases=np.stack(canvases,axis=0).transpose(1,2,0)
        data=np.memmap(wd+'\\data.npy',dtype=np.float32,mode='w+',shape=(images,boundry,boundry,3))
        mask=np.memmap(wd+'\\mask.npy',dtype=np.float32,mode='w+',shape=(images,boundry,boundry,len(forms)+1))
        for image in range(images):
            pos_c=random(c_size-boundry)
            pos_v=random(c_size-boundry)
            data[image]=canvas[pos_c:pos_c+boundry,pos_v:pos_v+boundry]
            mask[image]=canvases[pos_c:pos_c+boundry,pos_v:pos_v+boundry]
        label=[np.array(['Number_'+str(x+1) for x in range(len(data))]),np.array(['triangle','star','square','circle','background'])]
        if draw==True:
            import matplotlib.pyplot as plt
            plt.axis('off')
            plt.imshow(canvas,vmin=0,vmax=1)
            self.save_img(plt,'patterns')
        return data,mask,label
   
    
    def fix_data(self,split=[6,1,2]):# If data wasn't not specified in constructor file name will be used to process data. Data is normalized masks are fixed to value 1 and data is split to specifications.
        np.random.seed(42)
        if type(self.data)==str:
            from PIL import Image
            print('xx')
            data=[]
            for path in [self.data,self.mask]:
                files=os.listdir(path)
                data.append([np.asarray(Image.open(os.path.join(path,i))) for i in files])
            self.data,self.mask=data        
            self.mask=np.where(np.stack([(self.mask[i]-np.min(self.mask[i]))/(np.max(self.mask[i])-np.min(self.mask[i])) if np.max(self.mask[i])!=0 else self.mask[i] for i in range(len(self.mask))],axis=0)!=0,1,0) 
            print(np.min(self.mask),np.max(self.mask),np.mean(self.mask))
            self.data=np.stack([(self.data[i]-np.min(self.data[i]))/(np.max(self.data[i])-np.min(self.data[i])) if np.max(self.data[i])!=0 else self.data[i] for i in range(len(self.data))],axis=0)
            print(np.min(self.data),np.max(self.data),np.mean(self.data))
        elif self.data is None:
            self.data,self.mask,self.label=self.patterns()#uses pattern generator if no data were supplied!!!
        self.image_size=self.data.shape
        if len(self.data)!=len(self.label[0]):
            self.label=[]
            self.label.append(np.array([str(i) for i in range(len(self.data))]))
            self.label.append(np.array([str(i) for i in range(self.depth)]))
        print(type(self.data))
        length=np.arange(len(self.data))
        np.random.shuffle(length)
        indices=[]
        x=0
        for i in split:
            y=int(np.floor(x+len(length)/np.sum(split)*i))
            indices.append(length[x:y])
            x=y
        u=np.sum([len(x) for x in indices])-len(length)
        if u!=0:
            indices[-1]=np.append(indices[-1],length[u:])
        self.train_data,self.valid_data,self.test_data=[self.data[i] for i in indices]
        self.train_mask,self.valid_mask,self.test_mask=[self.mask[i] for i in indices]
        self.data_label,self.valid_label,self.test_label=[self.label[0][i] for i in indices]
        self.depth=self.mask.shape[-1]
        return self
    
    
    def down_stream(self): # Generates procedual downsampling step of NN. targed size and input shape are taken form constructor's values.
        #image=np.zeros(self.image_size,dtype=np.float32)
        self.steps=range(int(np.log2(self.step_size[1])-np.log2(self.step_size[0])))
        self.input=into=Input(self.image_size[1:])
        for x in self.steps:
            down=Conv2D(2**int(np.log2(self.step_size[0])+x),(3,3),activation='relu',padding='same')(into)
            #down=Dropout(0.2)(down)
            down=Conv2D(2**int(np.log2(self.step_size[0])+x),(3,3),activation='relu',padding='same')(down)
            self.transfer.append(down)
            into=MaxPooling2D((2,2))(down)
        return into
        
        
    def up_stream(self,into): #Generates procedual upsampling step of NN in reverse direction. Additionaly Conv2DTranspose layers are used to learn umpsampling steps and dropout function for regularization. 
        for x in reversed(self.steps[1:]):
            layerdepth=2**int(np.log2(self.step_size[0])+x)
            up=Conv2DTranspose(2**int(np.log2(self.step_size[0])+x),(1,1),strides=(2,2))(into)
            up=Conv2D(2**int(np.log2(self.step_size[0])+x), (3,3), activation = 'relu', padding = 'same')(up)#(UpSampling2D(size = (2,2))(into))
            merge = concatenate([self.transfer[x],up])#<--axis???
            up = Conv2D(2**int(np.log2(self.step_size[0])+x), (3,3), activation = 'relu', padding = 'same')(merge)
            up=Dropout(0.2)(up)
            into = Conv2D(2**int(np.log2(self.step_size[0])+x), (3,3), activation = 'relu', padding = 'same')(up)
        up=Conv2DTranspose(int(self.step_size[0]),(1,1),strides=(2,2))(into)
        up=Conv2D(int(self.step_size[0]),(3,3),activation='relu',padding='same')(up)#(UpSampling2D(size = (2,2))(into))
        merge = concatenate([self.transfer[0],up])
        up=Conv2D(int(self.step_size[0]),(3,3),activation='relu',padding='same')(merge)
        self.output=output=Conv2D(self.depth,(1,1),activation='softmax',padding='same')(up)
        model=Model(self.input,self.output)
        return model    
            
        
    def bottelneck(self,into): #bottleneck layer reverses direction form down- to up-stream.
        reverse = Conv2D(self.conv_depth, 3, activation = 'relu', padding = 'same')(into)
        #reverse = Dropout(0.2)(reverse)
        into = Conv2D(self.conv_depth, 3, activation = 'relu', padding = 'same')(reverse)
        return into
    
    
    def build_model(self,lr=0.001,path=None):#procedually creates model. Adam is used as SGD optimizer and he_normal as weight initializer or weights of previously trained model can be used. Metric is accuracy and learning rate is choosable
        self.model=self.up_stream(self.bottelneck(self.down_stream()))
        adam=optimizer.Adam(learning_rate=lr)
        for layer in self.model.layers:
            layer.kernel_initializer='he_normal'
        self.model.compile(optimizer=adam, loss="categorical_crossentropy",metrics=['acc'])
        self.model.summary()
        self.load_weights(path)
        return self
    
    def scheduler(self,epoch,lr,x=-.1):#adaptiv learning rate. reduces step size every ten epochs by e to the power of negativ x 
        if epoch%10!=0:
            return lr
        else:
            return lr * np.exp(x)
    
    def train(self,epochs=16,batch_size=16,safe=True,path=wd+'\\backup\\',data=None,mask=None,draw=False): #training routine. Either saveing model or not. default data is class training data.        
        if data is None:
            data=self.train_data
            mask=self.train_mask
        data_gen=ImageDataGenerator()
        valid_gen=ImageDataGenerator()
        test_gen=ImageDataGenerator()
        training_data=data_gen.flow(data,mask,batch_size=batch_size)
        valid=valid_gen.flow(self.valid_data,self.valid_mask,batch_size=batch_size)
        test_data=data_gen.flow(self.test_data,self.test_mask,batch_size=1)
        self.history=History()
        lr_update=LearningRateScheduler(self.scheduler)
        if safe==True:
            if os.path.isdir(path)!=True:
                try:
                    os.mkdir(os.path.join(path))
                except:
                    path=input('Enter backup directory:')
                    os.mkdir(os.path.join(path))
            cp_callback = ModelCheckpoint(filepath=path+'backup1.ckpt',
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             verbose=1)
            self.model.fit_generator(generator=training_data,
                           steps_per_epoch=len(data)/batch_size, 
                          validation_data=valid, 
                          validation_steps=len(self.valid_data)/batch_size,
                          epochs=epochs,
                                    callbacks=[cp_callback,lr_update,self.history])
        else:
            self.model.fit_generator(generator=training_data,
                       steps_per_epoch=len(data)/batch_size, 
                      validation_data=valid, 
                      validation_steps=len(self.valid_data)/batch_size,
                      epochs=epochs,
                      callbacks=[lr_update,self.history])
        if draw==True:
            import matplotlib.pyplot as plt
            acc=[u for u in self.history.history if 'acc' in u]
            for a in acc:
                plt.plot(np.arange(1,epochs+1,1),self.history.history[a])
            plt.title('Graph')
            plt.ylabel('Accuracy')
            plt.xlabel('epoch')
            plt.legend(acc, loc='upper left')
            self.save_img(plt,'graph')
            plt.show()
            
            
    def predict(self,images=None, mask=None, size=3, draw=True): # predict method is standard keras predict with option of drawing random samples.
        if images is None:
            images=self.test_data
            mask=self.test_mask
        if mask is None:
            mask=[s_mask for s_mask in [self.train_mask,self.valid_mask,self.test_mask] if len(s_mask)==len(images)][0]
        predictions=[]
        if size>len(images):
            size=len(images)
        for image in images:
            image=np.expand_dims(image, axis=0)
            predictions.append(self.model.predict(image))
        if draw==True:
            import matplotlib.pyplot as plt
            indizes=np.unique(np.random.randint(len(predictions),size=size))
            fig=plt.figure(figsize=(28,10*len(indizes)))
            pos_x=1
            line=self.depth
            for i in indizes:
                for k in range(2):
                    if (k+1)%2==0:
                        img=[np.squeeze(predictions[i])[:,:,u] for u in range(line)]
                        test_against=np.array([np.squeeze(mask[i])[:,:,u] for u in range(line)])
                        label_values=[str(np.round((1-np.sum(np.abs(test_against[u]-img[u])/(img[u].shape[0]*img[u].shape[1])))*100,2))+'%' for u in range(line)]
                        label=['Image Num '+str(i)]
                        label.extend(label_values)
                    else:
                        img=[np.squeeze(mask[i])[:,:,u] for u in range(line)]
                        label=[u for u in [self.data_label,self.valid_label,self.test_label] if len(u)==len(images)][0]
                        label=[label[i]]
                        label.extend(self.label[1])
                    img_truth=[u for u in [self.test_data,self.valid_data,self.train_data] if len(u)==len(images)][0]
                    img_truth=[img_truth[i]]
                    img_truth.extend(img)
                    img=img_truth
                    for ii,pos in zip(img,range(len(img))):
                        sub=fig.add_subplot(2*np.ceil(len(indizes))+1, line+1, pos_x)
                        pos_x+=1
                        sub.axis('off')
                        sub.set_title(label[pos])
                        sub.imshow(ii,vmin=0,vmax=1)
            self.save_img(fig,figname='predict')
            plt.show()
            
        loss, acc = self.model.evaluate(images,  mask, verbose=2)
        return np.concatenate(predictions,axis=0)
    
    
    def selectiv_(self,data,mask,epsilon=.01,arg_min=True,against_mask=False): # main algorithm for generating augmented training data generates all possible combination of postive negativ or neutral pertubation choosing the one with the highest accuracy score to add to trainings set.
        string_print='[ ] 0.00%'
        print(string_print,end='\r')
        targed_class_mask=[]
        if len(data.shape)==3:
            np.expand_dims(data, axis=0)
        if len(mask.shape)==3:
            np.expand_dims(mask, axis=0)
        for xy in range(self.depth):
            class_mask=np.zeros(mask.shape[1:])
            class_mask[:,:,xy]+=np.max(mask)
            targed_class_mask.append(class_mask)
        results=np.zeros(shape=data.shape,dtype=np.float32)
        hh=0
        width=self.depth
        for x,y in zip(data,mask):
            result=[]
            patterns=[]
            zeros=[np.zeros(data[:1].shape)]*width
            patterns.append(zeros)
            if against_mask==True:
                for targed,u in zip(targed_class_mask,range(width)):
                    targed[:,:,u]=y[:,:,u]
            for i in [-1,1]:
                pattern=[np.array(self.create_adversarial_pattern(np.expand_dims(x,axis=0),yy)*i*epsilon) for yy in targed_class_mask]# adverserial pertubation generator
                patterns.append(pattern)
            element=[]
            iters=len(patterns)
            pos=max([len(i) for i in patterns])
            dim=[0]*pos
            point='|'
            for poss in range(iters**pos*pos):# iteration logic 
                line=poss%pos
                #point=('|' if point=='-' else '-' )
                #print(string_print+point,end='\r')
                element.extend(patterns[dim[line]][line])
                if line==pos-1:
                    condendor=np.clip(x+sum(element),0,1)# staking perturbation
                    result.append(condendor)
                    element=[]
                    for i in range(pos):
                        if dim[i]!=iters-1:
                            dim[i]+=1
                            break
                        else:
                            dim[i]=0
            if arg_min==True:# choice of highest accuracy
                results[hh]=result[np.argmin([np.mean(np.abs(y-self.model.predict(np.expand_dims(condendor,axis=0)))) for condendor in result])]
            else:
                results[hh]=result[np.argmax([np.mean(np.abs(y-self.model.predict(np.expand_dims(condendor,axis=0)))) for condendor in result])]
            hh+=1
            how_far=hh/len(data)
            string_print='['+'||'*int(np.floor(how_far*10))+'] '+str(round(how_far*100,2))+'%'
            print(string_print,end='  \r')
        print('\nDone!')
        return results
    
    def create_adversarial_pattern(self,input_image, mask_image):#vanilla FGSM 
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        shape=list(mask_image.shape)#does basically nothing
        input_image=tf.cast(input_image, tf.float32)
        label=tf.cast(mask_image, tf.float32)
        pretrained_model=self.model
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = pretrained_model(input_image)
            loss = loss_object(label, prediction)
            gradient = tape.gradient(loss, input_image)
            signed_grad = tf.sign(gradient)
        return signed_grad
    
    
    def create_adv_image(self, input_image, mask_image,Un_Target,epsilon=.01):#adds FGSM pattern to image and clips it to value space
        add=Un_Target
        input_image=np.clip(input_image+self.create_adversarial_pattern(input_image, mask_image)*epsilon*add,0,1)
        return input_image
    
    
    def create_iterativ_adv_image(self, input_image, mask_image,Un_Target,epsilon=.001, iters=100):#basically PGD. 100 iterations is a lot
        add=Un_Target
        for i in range(iters):#This is where it becomes slow
            input_image=np.clip(input_image+self.create_adversarial_pattern(input_image, mask_image)*epsilon*add,0,1)
        return input_image
        
        
    def evaluate_adv(self,data=None, mask=None, epsilon=.001, class_to=None, add=-1, iterativ=True, draw=False):#All pertubation methods combined in one method(FGSM PGD). Choice of Ascent or descent as well as target class. Option for draw random samples.
        methode=[self.create_adv_image,self.create_iterativ_adv_image]
        if data is None or mask is None:
            data=self.test_data
            mask=self.test_mask
        mask_shape=mask.shape
        width=mask_shape[-1]
        if class_to!=None:
            try:
                class_to=class_to-1
            except:
                class_to=int(input('Please enter a Number between 1 and '+str(width)))
                class_to=class_to-1
            targed_class_mask=np.zeros(mask_shape)
            targed_class_mask[:,:,:,class_to]+=np.max(mask)
        else:
            targed_class_mask=mask
        data_adv=methode[int(iterativ)](data,targed_class_mask,add,epsilon)
        if draw==True:
            import matplotlib.pyplot as plt
            clean_pred=self.model.predict(data)# having this outside the if clause was a realy stupid mistake
            adv=np.clip(np.abs(data_adv-data),0,1)
            adv_pred=self.model.predict(data_adv)                      
            graph_y=int(np.ceil(len(data)*20/100))
            indizes=np.unique(np.random.randint(len(data), size=graph_y))
            fig=plt.figure(figsize=(28,10*len(indizes)))
            pos_x=1
            for i in indizes:
                for x in range(3):
                    if (x+1)%3==0:
                        img_src=adv_pred[i]
                        src=data_adv[i]
                        title='Epsilon '+str(epsilon*100)+'%'
                    elif (x+1)%3==2:
                        img_src=clean_pred[i]
                        src=data[i]
                        title='Clean'
                    elif (x+1)%3==1:
                        img_src=mask[i]
                        src=np.clip(adv[i],0,1)
                        title='Target'
                    graph_x=img_src.shape[-1]+1
                    sub=fig.add_subplot(graph_y*3, graph_x , pos_x)
                    sub.axis('off')
                    sub.set_title(title)
                    sub.imshow(src,vmin=0,vmax=1)
                    pos_x+=1
                    proz=[str(round(np.mean(1-(np.abs(img_src[:,:,u]-mask[i,:,:,u]))/np.max(mask)),4)*100)+' %' for u in range(width)]
                    for ii in range(width):
                        sub=fig.add_subplot(graph_y*3, graph_x , pos_x)
                        sub.axis('off')
                        sub.set_title(str(proz[ii]))
                        sub.imshow(img_src[:,:,ii],vmin=0,vmax=1)
                        pos_x+=1
            self.save_img(fig,figname='evaluate')
            plt.show()
        return np.float32(data_adv), mask


    def distill_patterns(self, images, predictions, add, epsilon=.01, class_to=None, iterativ=True):#due to memory limitations feeding logic for evaluate_adv
        x=0
        step=15
        pre_arr=[]
        dis=len(predictions)
        print('[ ] 0.00%',end='\r')
        while x<dis:
            if dis-x>step:
                y=x+step
            else:
                y=dis
            pre_,_=self.evaluate_adv(images[x:y],predictions[x:y],epsilon,class_to,add,iterativ)
            pre_arr.extend(pre_)
            x=y
            how_far=x/dis
            string_print='['+'||'*int(np.floor(how_far*10))+'] '+str(round(how_far*100,2))+'%'
            print(string_print,end='\r')
        print('\nDone!')
        return np.stack(pre_arr,axis=0)


    def load_weights(self,path=None): # for transfer of model parameters from one model to another
        if path==None:
            path=input('Enter Checkpoint location: ')
            if path.lower().strip() in ['no','quit','exit']: 
                print("ok")
            else:
                path=os.path.normpath(path)
                try:
                    self.model.load_weights(path)
                    print('Model loaded!')
                    loss, acc = self.model.evaluate(self.test_data,  self.test_mask, verbose=2)
                except:
                    print('Failure! No pretrained weights were loaded')
        else:
                path=os.path.normpath(path)
                try:
                    self.model.load_weights(path)
                    print('Model loaded!')
                    loss, acc = self.model.evaluate(self.test_data,  self.test_mask, verbose=2)
                except:
                    print('Failure!')
      
    
    def load_data(self,data): #to except data from another class
        try:
            self.train_data,self.valid_data,self.test_data,self.train_mask,self.valid_mask,self.test_mask,self.data_label,self.valid_label,self.test_label,self.label=data
            self.image_size=self.train_data.shape
            self.depth=self.train_mask.shape[-1]
            print('loaded')
        except:
            print('Something went wrong')
        return self
    
    
    def return_data(self): #returns all data set relaited arrays
        try:
            return [self.train_data,self.valid_data,self.test_data,self.train_mask,self.valid_mask,self.test_mask,self.data_label,self.valid_label,self.test_label,self.label]
        except:
            print("Data is not yet generated")
    
    
    def return_model(self): #exports model from class
        try:
            return self.model
        except:
            print('model does not yet exist')


    def draw_random(self,data,to_pred,mask,in_pred,size=10): #draws random samples of presented data sets.
        import matplotlib.pyplot as plt
        if size>len(data):
            size=len(data)
        width=self.depth
        indizes=np.random.randint(len(in_pred),size=size)
        fig=plt.figure(figsize=(28,10*len(indizes)))
        pos_x=1
        for index in indizes:
            for k in range(2): 
                if (k+1)%2==1:
                    img=[data[index]]
                    target=[mask[index,:,:,i] for i in range(width)]
                    img.extend(target)
                    label=['Target']
                    label.extend(['100%']*width)
                else:
                    img=[to_pred[index]]
                    pred=[in_pred[index,:,:,i] for i in range(width)]
                    img.extend(pred)
                    label=['Rel.Diff.'+str(np.mean(np.abs(data[index]-to_pred[index])))]
                    label.extend([str(np.round(1-np.mean(np.abs(target[i]-pred[i])),4)*100)+'%' for i in range(width)])
                for ii,pos in zip(img,range(width+1)):
                    sub=fig.add_subplot(2*len(indizes)+1, width+1, pos_x)
                    pos_x+=1
                    sub.axis('off')
                    sub.set_title(label[pos])
                    sub.imshow(ii,vmin=0,vmax=1)
        self.save_img(fig,figname='random')
        plt.show()
        return self
    
    
    def save_img(self,fig,figname='random',path_pic=wd+'\\images\\'):#logic for saveing samples drawn with matplot lib
        if figname!='no':
            img_name=figname+'1.png'
            try:
                pics=os.listdir(path_pic)
            except:
                os.mkdir(path_pic)
                pics=[]
            if img_name not in pics: 
                fig.savefig(path_pic+img_name)
            else:
                num=1
                while img_name in pics:
                    img_name=figname+str(num)+'.png'
                    num+=1
                fig.savefig(path_pic+img_name)