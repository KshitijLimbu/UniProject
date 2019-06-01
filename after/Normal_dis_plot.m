close all ; clear ; clc ;
t r a i n i n g S a m p l e s = 1000;
t e s t i n g S a m p l e s = 200;
% [ trainingData , t r a i n i n g T a r g e t ] = G e n e r a t e N o n L i n e a r ( t r a i n i n g S a m p l e s ) ;
% [ testingData , t e s t i n g T a r g e t ] = G e n e r a t e N o n L i n e a r ( t e s t i n g S a m p l e s ) ;
% [ trainingData , t r a i n i n g T a r g e t ] = G e n e r a t e N o n L i n e a r 2 ( t r a i n i n g S a m p l e s ) ;
% [ testingData , t e s t i n g T a r g e t ] = G e n e r a t e N o n L i n e a r 2 ( t e s t i n g S a m p l e s ) ;
[ trainingData , t r a i n i n g T a r g e t ] = G e n e r a t e N o n L i n e a r 3 ( t r a i n i n g S a m p l e s ) ;
[ testingData , t e s t i n g T a r g e t ] = G e n e r a t e N o n L i n e a r 3 ( t e s t i n g S a m p l e s ) ;
% [ trainingData , t r a i n i n g T a r g e t ] = XOR ( t r a i n i n g S a m p l e s ) ;
% [ testingData , t e s t i n g T a r g e t ] = XOR ( t e s t i n g S a m p l e s ) ;
t r a i n i n g T a r g e t (2 ,:) = [];
t e s t i n g T a r g e t (2 ,:) = [];
% t h r e e _ l a y e r _ n e t w o r k _ t r a i n ( epochs , input , targets , no_of_output , no_hidden_neuron , output_function , error_function ,
algorithm , learning , momentum , learning_rate , error_val , mini_batch , mb_value , d i s t r i b u t i o n )
% t h r e e _ l a y e r
% [ first_layer , second_layer , error1 ] = t h r e e _ l a y e r _ n e t w o r k _ t r a i n (40 , trainingData ’ , trainingTarget ’ ,1 , 4 , ’ sigmoid ’ ,
’ cross entropy s ’ , ’ feed back_alignment ’ , ’ online ’ , 1 ,0.2 , 0.005 ,1 , 200 ,0.2) ;
% [ first_layer , second_layer , error2 ] = t h r e e _ l a y e r _ n e t w o r k _ t r a i n (40 , trainingData ’ , trainingTarget ’ ,1 , 4 , ’ sigmoid ’ ,
’ cross entropy s ’ , ’ feed back_alignment ’ , ’ online ’ , 1 ,0.2 , 0.005 ,1 , 200 ,0.5) ;
% [ first_layer , second_layer , error3 ] = t h r e e _ l a y e r _ n e t w o r k _ t r a i n (40 , trainingData ’ , trainingTarget ’ ,1 , 4 , ’ sigmoid ’ ,
’ cross entropy s ’ , ’ feed back_alignment ’ , ’ online ’ , 1 ,0.2 , 0.005 ,1 , 200 ,0.7) ;
% [ first_layer , second_layer , error4 ] = t h r e e _ l a y e r _ n e t w o r k _ t r a i n (40 , trainingData ’ , trainingTarget ’ ,1 , 4 , ’ sigmoid ’ ,
’ cross entropy s ’ , ’ feed back_alignment ’ , ’ online ’ , 1 ,0.2 , 0.005 ,1 , 200 ,0.9) ;
% m u l t i p l e _ l a y e r
net = [20 20];
% m u l t i p l e _ l a y e r _ n e t w o r k _ t r a i n ( epochs , input , targets , no_of_output , net_architecture , output_function ,
error_function , algorithm , learning , momentum , learning_rate , error_val , mb_value , partial_con ,
d i s t r i b u t i o n )
[ weight , error1 ] = m u l t i p l e _ l a y e r _ n e t w o r k _ t r a i n (80 , trainingData ’ , trainingTarget ’ , 1 , net , ’ sigmoid ’ , ’ cross entropy
s ’ , ’ f e e d b a c k _ a l i g n m e n t ’ , ’ online ’ ,1 , 0.2 , 0.005 ,200 ,0 ,0.2) ;
[ weight , error2 ] = m u l t i p l e _ l a y e r _ n e t w o r k _ t r a i n (80 , trainingData ’ , trainingTarget ’ , 1 , net , ’ sigmoid ’ , ’ cross entropy
s ’ , ’ f e e d b a c k _ a l i g n m e n t ’ , ’ online ’ ,1 , 0.2 , 0.005 ,200 ,0 ,0.5) ;
[ weight , error3 ] = m u l t i p l e _ l a y e r _ n e t w o r k _ t r a i n (80 , trainingData ’ , trainingTarget ’ , 1 , net , ’ sigmoid ’ , ’ cross entropy
s ’ , ’ f e e d b a c k _ a l i g n m e n t ’ , ’ online ’ ,1 , 0.2 , 0.005 ,200 ,0 ,0.7) ;
[ weight , error4 ] = m u l t i p l e _ l a y e r _ n e t w o r k _ t r a i n (80 , trainingData ’ , trainingTarget ’ , 1 , net , ’ sigmoid ’ , ’ cross entropy
s ’ , ’ f e e d b a c k _ a l i g n m e n t ’ , ’ online ’ ,1 , 0.2 , 0.005 ,200 ,0 ,0.9) ;
figure
x =1:1:65;
plot (x , error1 ’ ,x , error2 ’ ,x , error3 ’ ,x , error4 ’) ;
title ( ’ error ’) ;
legend ( ’ var = 0.2 ’ , ’ var = 0.5 ’ , ’ var = 0.7 ’ , ’ var = 0.9 ’) ;
