import pandas as pd

import math 


class Neurons:


    


    def __init__(self,av,num_weights,index_neuron):
        self.av=av
        self.num_weights=num_weights
        self.index_neuron=index_neuron
        self.weights=[]
        for i in range(self.num_weights):
            self.weights.append(0.0)
        
    def multweights(self,prev_layer):#sigmoid function
        result=0
        #weights=[]
        for i in range(len(prev_layer)):
            if i == 2:
                result=result+prev_layer[i].weights[self.index_neuron]
            else:
                result=result+((float((prev_layer[i]).av))*prev_layer[i].weights[self.index_neuron])
        
        av=(1/(1+math.exp(-1*0.5*(result))))
        return av
    
    
input_layer=[Neurons(0,4,0),Neurons(0,4,1),Neurons(1,4,2)] # 1st argument is the initial activation value(will be updated in feedforward function
                                            # 2nd is the number of weights, defines how many nodes in next layer
                                            # 3rd is index, gives the index nuber(row number)
                                            #input_layer[2] is the bias


hidden_layer=[Neurons(0, 2, 0), Neurons(0, 2, 1),Neurons(0, 2, 2),Neurons(0, 2, 3), Neurons(1, 2, 4)]

output_layer=[Neurons(0, 0, 0),Neurons(0, 0, 1)]

weights=pd.read_csv("weights")





input_layer[0].weights[0]=weights.values[0,0]



input_layer[0].weights[1]=weights.values[1,0]

input_layer[0].weights[2]=weights.values[2,0]

input_layer[0].weights[3]=weights.values[3,0]

print(input_layer[0].weights)

input_layer[1].weights[0]=weights.values[4,0]

input_layer[1].weights[1]=weights.values[5,0]

input_layer[1].weights[2]=weights.values[6,0]

input_layer[1].weights[3]=weights.values[7,0]

input_layer[2].weights[0]=weights.values[8,0]

input_layer[2].weights[1]=weights.values[9,0]

input_layer[2].weights[2]=weights.values[10,0]

input_layer[2].weights[3]=weights.values[11,0]


hidden_layer[0].weights[0]=weights.values[0,1]

hidden_layer[0].weights[1]=weights.values[1,1]

hidden_layer[1].weights[0]=weights.values[2,1]

hidden_layer[1].weights[1]=weights.values[3,1]

hidden_layer[2].weights[0]=weights.values[4,1]

hidden_layer[2].weights[1]=weights.values[5,1]

hidden_layer[3].weights[0]=weights.values[6,1]

hidden_layer[3].weights[1]=weights.values[7,1]

hidden_layer[4].weights[0]=weights.values[8,1]

hidden_layer[4].weights[1]=weights.values[9,1]



#print(weights)


def feedforward_process(data):
    predicted_output=[]
    for i in range(len(input_layer)):
        if i == 2:
            input_layer[i].av = 1
        else:
            input_layer[i].av = data[i]
            #print("input_layer",i,"av =",input_layer[i].av)

    for i in range(len(hidden_layer)-1):
        if i == 2:
            hidden_layer[i].av = 1
        else:
            hidden_layer[i].av=hidden_layer[i].multweights(input_layer)
            #print("Hidden layer",i,"av i ",hidden_layer[i].av)
    for i in range(len(output_layer)):
        output_layer[i].av=output_layer[i].multweights(hidden_layer)
        predicted_output.append(output_layer[i].av)
        #print("av of output laye",i, "is", output_layer[i].av)
        
   # print(predicted_output)
        
    return predicted_output


maximum_of_x1=543.4653405
minimum_of_x1=-589.5854999
maximum_of_x2=689.3468253
minimum_of_x2=65.93032699
maximum_of_y1=7.977472435
minimum_of_y1=-2.865481767
maximum_of_y2=4.575297258
minimum_of_y2=-4.490872799







class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        
        


        
        



        


        
        
        
        

    
    def predict(self, input_row):
        
        
        print("csv =",input_row) 
        input_rows=input_row.split(',')
        for i in range(len(input_rows)):
              input_rows[i]=float(input_rows[i])
        
        
        if input_rows[0]<=0:
            input_rows[0]= ((input_rows[0]-minimum_of_x1)/(maximum_of_x1- minimum_of_x1))-0.65 # 0.65 is the error on our output which was calculated after various trial and error
            
        
            
        else:
            input_rows[0]= ((input_rows[0]-minimum_of_x1)/(maximum_of_x1- minimum_of_x1))+0.65
            
        
        
        
        if input_rows[1]==0:
            input_rows[1]=0
        else:
            input_rows[1]= (input_rows[1]-minimum_of_x2)/(maximum_of_x2- minimum_of_x2)
            
        print("normalised 0 =",input_rows[0])
        print("normalised 1 =",input_rows[1])
        output=[]
        output=feedforward_process(input_rows)
        output[0]=0.4*(((output[0]*(maximum_of_y1-minimum_of_y1))+minimum_of_y1))
        output[1]=-0.1*((output[1]*(maximum_of_y2-minimum_of_y2))+minimum_of_y2)
        
        print("denormalised Output =",output)
        
            
        return output
      
