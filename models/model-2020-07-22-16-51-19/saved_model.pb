��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02unknown8��
�
conv_model/conv_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv_model/conv_1_1/kernel
�
.conv_model/conv_1_1/kernel/Read/ReadVariableOpReadVariableOpconv_model/conv_1_1/kernel*&
_output_shapes
: *
dtype0
�
conv_model/conv_1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv_model/conv_1_1/bias
�
,conv_model/conv_1_1/bias/Read/ReadVariableOpReadVariableOpconv_model/conv_1_1/bias*
_output_shapes
: *
dtype0
�
conv_model/conv_1_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *+
shared_nameconv_model/conv_1_2/kernel
�
.conv_model/conv_1_2/kernel/Read/ReadVariableOpReadVariableOpconv_model/conv_1_2/kernel*&
_output_shapes
:  *
dtype0
�
conv_model/conv_1_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv_model/conv_1_2/bias
�
,conv_model/conv_1_2/bias/Read/ReadVariableOpReadVariableOpconv_model/conv_1_2/bias*
_output_shapes
: *
dtype0
�
conv_model/conv_2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv_model/conv_2_1/kernel
�
.conv_model/conv_2_1/kernel/Read/ReadVariableOpReadVariableOpconv_model/conv_2_1/kernel*&
_output_shapes
: @*
dtype0
�
conv_model/conv_2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv_model/conv_2_1/bias
�
,conv_model/conv_2_1/bias/Read/ReadVariableOpReadVariableOpconv_model/conv_2_1/bias*
_output_shapes
:@*
dtype0
�
conv_model/conv_2_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*+
shared_nameconv_model/conv_2_2/kernel
�
.conv_model/conv_2_2/kernel/Read/ReadVariableOpReadVariableOpconv_model/conv_2_2/kernel*&
_output_shapes
:@@*
dtype0
�
conv_model/conv_2_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv_model/conv_2_2/bias
�
,conv_model/conv_2_2/bias/Read/ReadVariableOpReadVariableOpconv_model/conv_2_2/bias*
_output_shapes
:@*
dtype0
�
conv_model/logit/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�T*(
shared_nameconv_model/logit/kernel
�
+conv_model/logit/kernel/Read/ReadVariableOpReadVariableOpconv_model/logit/kernel*
_output_shapes
:	�T*
dtype0
�
conv_model/logit/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv_model/logit/bias
{
)conv_model/logit/bias/Read/ReadVariableOpReadVariableOpconv_model/logit/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
�%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�$
value�$B�$ B�$
�
conv_1_1
conv_1_2

max_pool_1
conv_2_1
conv_2_2

max_pool_2
flatten
	dense
		optimizer

_training_endpoints
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
R
,regularization_losses
-	variables
.trainable_variables
/	keras_api
R
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
6
:iter
	;decay
<learning_rate
=momentum
 
 
F
0
1
2
3
 4
!5
&6
'7
48
59
F
0
1
2
3
 4
!5
&6
'7
48
59
�
>non_trainable_variables

?layers
regularization_losses
@metrics
Alayer_regularization_losses
trainable_variables
	variables
 
ZX
VARIABLE_VALUEconv_model/conv_1_1/kernel*conv_1_1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_model/conv_1_1/bias(conv_1_1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Bnon_trainable_variables

Clayers
regularization_losses
Dmetrics
	variables
trainable_variables
Elayer_regularization_losses
ZX
VARIABLE_VALUEconv_model/conv_1_2/kernel*conv_1_2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_model/conv_1_2/bias(conv_1_2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Fnon_trainable_variables

Glayers
regularization_losses
Hmetrics
	variables
trainable_variables
Ilayer_regularization_losses
 
 
 
�
Jnon_trainable_variables

Klayers
regularization_losses
Lmetrics
	variables
trainable_variables
Mlayer_regularization_losses
ZX
VARIABLE_VALUEconv_model/conv_2_1/kernel*conv_2_1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_model/conv_2_1/bias(conv_2_1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
�
Nnon_trainable_variables

Olayers
"regularization_losses
Pmetrics
#	variables
$trainable_variables
Qlayer_regularization_losses
ZX
VARIABLE_VALUEconv_model/conv_2_2/kernel*conv_2_2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_model/conv_2_2/bias(conv_2_2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
�
Rnon_trainable_variables

Slayers
(regularization_losses
Tmetrics
)	variables
*trainable_variables
Ulayer_regularization_losses
 
 
 
�
Vnon_trainable_variables

Wlayers
,regularization_losses
Xmetrics
-	variables
.trainable_variables
Ylayer_regularization_losses
 
 
 
�
Znon_trainable_variables

[layers
0regularization_losses
\metrics
1	variables
2trainable_variables
]layer_regularization_losses
TR
VARIABLE_VALUEconv_model/logit/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEconv_model/logit/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
�
^non_trainable_variables

_layers
6regularization_losses
`metrics
7	variables
8trainable_variables
alayer_regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

b0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	ctotal
	dcount
e
_fn_kwargs
fregularization_losses
g	variables
htrainable_variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

c0
d1
 
�
jnon_trainable_variables

klayers
fregularization_losses
lmetrics
g	variables
htrainable_variables
mlayer_regularization_losses

c0
d1
 
 
 
�
serving_default_input_1Placeholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv_model/conv_1_1/kernelconv_model/conv_1_1/biasconv_model/conv_1_2/kernelconv_model/conv_1_2/biasconv_model/conv_2_1/kernelconv_model/conv_2_1/biasconv_model/conv_2_2/kernelconv_model/conv_2_2/biasconv_model/logit/kernelconv_model/logit/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_134249
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.conv_model/conv_1_1/kernel/Read/ReadVariableOp,conv_model/conv_1_1/bias/Read/ReadVariableOp.conv_model/conv_1_2/kernel/Read/ReadVariableOp,conv_model/conv_1_2/bias/Read/ReadVariableOp.conv_model/conv_2_1/kernel/Read/ReadVariableOp,conv_model/conv_2_1/bias/Read/ReadVariableOp.conv_model/conv_2_2/kernel/Read/ReadVariableOp,conv_model/conv_2_2/bias/Read/ReadVariableOp+conv_model/logit/kernel/Read/ReadVariableOp)conv_model/logit/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_134350
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_model/conv_1_1/kernelconv_model/conv_1_1/biasconv_model/conv_1_2/kernelconv_model/conv_1_2/biasconv_model/conv_2_1/kernelconv_model/conv_2_1/biasconv_model/conv_2_2/kernelconv_model/conv_2_2/biasconv_model/logit/kernelconv_model/logit/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcount*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_134410�
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_134255

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@*  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������T2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������T2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
G
+__inference_max_pool_2_layer_call_fn_134147

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_max_pool_2_layer_call_and_return_conditional_losses_1341412
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�*
�
__inference__traced_save_134350
file_prefix9
5savev2_conv_model_conv_1_1_kernel_read_readvariableop7
3savev2_conv_model_conv_1_1_bias_read_readvariableop9
5savev2_conv_model_conv_1_2_kernel_read_readvariableop7
3savev2_conv_model_conv_1_2_bias_read_readvariableop9
5savev2_conv_model_conv_2_1_kernel_read_readvariableop7
3savev2_conv_model_conv_2_1_bias_read_readvariableop9
5savev2_conv_model_conv_2_2_kernel_read_readvariableop7
3savev2_conv_model_conv_2_2_bias_read_readvariableop6
2savev2_conv_model_logit_kernel_read_readvariableop4
0savev2_conv_model_logit_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_93585a98e1fc4d9a8bdd31f804c8e3f3/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B*conv_1_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB(conv_1_1/bias/.ATTRIBUTES/VARIABLE_VALUEB*conv_1_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB(conv_1_2/bias/.ATTRIBUTES/VARIABLE_VALUEB*conv_2_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB(conv_2_1/bias/.ATTRIBUTES/VARIABLE_VALUEB*conv_2_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB(conv_2_2/bias/.ATTRIBUTES/VARIABLE_VALUEB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_conv_model_conv_1_1_kernel_read_readvariableop3savev2_conv_model_conv_1_1_bias_read_readvariableop5savev2_conv_model_conv_1_2_kernel_read_readvariableop3savev2_conv_model_conv_1_2_bias_read_readvariableop5savev2_conv_model_conv_2_1_kernel_read_readvariableop3savev2_conv_model_conv_2_1_bias_read_readvariableop5savev2_conv_model_conv_2_2_kernel_read_readvariableop3savev2_conv_model_conv_2_2_bias_read_readvariableop2savev2_conv_model_logit_kernel_read_readvariableop0savev2_conv_model_logit_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : :  : : @:@:@@:@:	�T:: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
�
+__inference_conv_model_layer_call_fn_134227
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv_model_layer_call_and_return_conditional_losses_1342112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������@@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
)__inference_conv_1_2_layer_call_fn_134081

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv_1_2_layer_call_and_return_conditional_losses_1340732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
A__inference_logit_layer_call_and_return_conditional_losses_134271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������T::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
&__inference_logit_layer_call_fn_134278

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_logit_layer_call_and_return_conditional_losses_1341972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������T::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
D__inference_conv_2_2_layer_call_and_return_conditional_losses_134127

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
D__inference_conv_1_1_layer_call_and_return_conditional_losses_134052

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_134177

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@*  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������T2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������T2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
b
F__inference_max_pool_2_layer_call_and_return_conditional_losses_134141

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�0
�
F__inference_conv_model_layer_call_and_return_conditional_losses_134211
input_1+
'conv_1_1_statefulpartitionedcall_args_1+
'conv_1_1_statefulpartitionedcall_args_2+
'conv_1_2_statefulpartitionedcall_args_1+
'conv_1_2_statefulpartitionedcall_args_2+
'conv_2_1_statefulpartitionedcall_args_1+
'conv_2_1_statefulpartitionedcall_args_2+
'conv_2_2_statefulpartitionedcall_args_1+
'conv_2_2_statefulpartitionedcall_args_2(
$logit_statefulpartitionedcall_args_1(
$logit_statefulpartitionedcall_args_2
identity�� conv_1_1/StatefulPartitionedCall� conv_1_2/StatefulPartitionedCall� conv_2_1/StatefulPartitionedCall� conv_2_2/StatefulPartitionedCall�logit/StatefulPartitionedCall�
 conv_1_1/StatefulPartitionedCallStatefulPartitionedCallinput_1'conv_1_1_statefulpartitionedcall_args_1'conv_1_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������>> *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv_1_1_layer_call_and_return_conditional_losses_1340522"
 conv_1_1/StatefulPartitionedCall�
conv_1_1/IdentityIdentity)conv_1_1/StatefulPartitionedCall:output:0!^conv_1_1/StatefulPartitionedCall*
T0*/
_output_shapes
:���������>> 2
conv_1_1/Identity�
 conv_1_2/StatefulPartitionedCallStatefulPartitionedCallconv_1_1/Identity:output:0'conv_1_2_statefulpartitionedcall_args_1'conv_1_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������<< *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv_1_2_layer_call_and_return_conditional_losses_1340732"
 conv_1_2/StatefulPartitionedCall�
conv_1_2/IdentityIdentity)conv_1_2/StatefulPartitionedCall:output:0!^conv_1_2/StatefulPartitionedCall*
T0*/
_output_shapes
:���������<< 2
conv_1_2/Identity�
max_pool_1/PartitionedCallPartitionedCallconv_1_2/Identity:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_max_pool_1_layer_call_and_return_conditional_losses_1340872
max_pool_1/PartitionedCall�
max_pool_1/IdentityIdentity#max_pool_1/PartitionedCall:output:0*
T0*/
_output_shapes
:��������� 2
max_pool_1/Identity�
 conv_2_1/StatefulPartitionedCallStatefulPartitionedCallmax_pool_1/Identity:output:0'conv_2_1_statefulpartitionedcall_args_1'conv_2_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv_2_1_layer_call_and_return_conditional_losses_1341062"
 conv_2_1/StatefulPartitionedCall�
conv_2_1/IdentityIdentity)conv_2_1/StatefulPartitionedCall:output:0!^conv_2_1/StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2
conv_2_1/Identity�
 conv_2_2/StatefulPartitionedCallStatefulPartitionedCallconv_2_1/Identity:output:0'conv_2_2_statefulpartitionedcall_args_1'conv_2_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv_2_2_layer_call_and_return_conditional_losses_1341272"
 conv_2_2/StatefulPartitionedCall�
conv_2_2/IdentityIdentity)conv_2_2/StatefulPartitionedCall:output:0!^conv_2_2/StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2
conv_2_2/Identity�
max_pool_2/PartitionedCallPartitionedCallconv_2_2/Identity:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_max_pool_2_layer_call_and_return_conditional_losses_1341412
max_pool_2/PartitionedCall�
max_pool_2/IdentityIdentity#max_pool_2/PartitionedCall:output:0*
T0*/
_output_shapes
:���������@2
max_pool_2/Identity�
flatten/PartitionedCallPartitionedCallmax_pool_2/Identity:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������T*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1341772
flatten/PartitionedCall�
flatten/IdentityIdentity flatten/PartitionedCall:output:0*
T0*(
_output_shapes
:����������T2
flatten/Identity�
logit/StatefulPartitionedCallStatefulPartitionedCallflatten/Identity:output:0$logit_statefulpartitionedcall_args_1$logit_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_logit_layer_call_and_return_conditional_losses_1341972
logit/StatefulPartitionedCall�
logit/IdentityIdentity&logit/StatefulPartitionedCall:output:0^logit/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2
logit/Identity�
IdentityIdentitylogit/Identity:output:0!^conv_1_1/StatefulPartitionedCall!^conv_1_2/StatefulPartitionedCall!^conv_2_1/StatefulPartitionedCall!^conv_2_2/StatefulPartitionedCall^logit/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������@@::::::::::2D
 conv_1_1/StatefulPartitionedCall conv_1_1/StatefulPartitionedCall2D
 conv_1_2/StatefulPartitionedCall conv_1_2/StatefulPartitionedCall2D
 conv_2_1/StatefulPartitionedCall conv_2_1/StatefulPartitionedCall2D
 conv_2_2/StatefulPartitionedCall conv_2_2/StatefulPartitionedCall2>
logit/StatefulPartitionedCalllogit/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�	
�
A__inference_logit_layer_call_and_return_conditional_losses_134197

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������T::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_134249
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_1340392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������@@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�B
�
!__inference__wrapped_model_134039
input_16
2conv_model_conv_1_1_conv2d_readvariableop_resource7
3conv_model_conv_1_1_biasadd_readvariableop_resource6
2conv_model_conv_1_2_conv2d_readvariableop_resource7
3conv_model_conv_1_2_biasadd_readvariableop_resource6
2conv_model_conv_2_1_conv2d_readvariableop_resource7
3conv_model_conv_2_1_biasadd_readvariableop_resource6
2conv_model_conv_2_2_conv2d_readvariableop_resource7
3conv_model_conv_2_2_biasadd_readvariableop_resource3
/conv_model_logit_matmul_readvariableop_resource4
0conv_model_logit_biasadd_readvariableop_resource
identity��*conv_model/conv_1_1/BiasAdd/ReadVariableOp�)conv_model/conv_1_1/Conv2D/ReadVariableOp�*conv_model/conv_1_2/BiasAdd/ReadVariableOp�)conv_model/conv_1_2/Conv2D/ReadVariableOp�*conv_model/conv_2_1/BiasAdd/ReadVariableOp�)conv_model/conv_2_1/Conv2D/ReadVariableOp�*conv_model/conv_2_2/BiasAdd/ReadVariableOp�)conv_model/conv_2_2/Conv2D/ReadVariableOp�'conv_model/logit/BiasAdd/ReadVariableOp�&conv_model/logit/MatMul/ReadVariableOp�
)conv_model/conv_1_1/Conv2D/ReadVariableOpReadVariableOp2conv_model_conv_1_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)conv_model/conv_1_1/Conv2D/ReadVariableOp�
conv_model/conv_1_1/Conv2DConv2Dinput_11conv_model/conv_1_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>> *
paddingVALID*
strides
2
conv_model/conv_1_1/Conv2D�
*conv_model/conv_1_1/BiasAdd/ReadVariableOpReadVariableOp3conv_model_conv_1_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv_model/conv_1_1/BiasAdd/ReadVariableOp�
conv_model/conv_1_1/BiasAddBiasAdd#conv_model/conv_1_1/Conv2D:output:02conv_model/conv_1_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>> 2
conv_model/conv_1_1/BiasAdd�
conv_model/conv_1_1/ReluRelu$conv_model/conv_1_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������>> 2
conv_model/conv_1_1/Relu�
)conv_model/conv_1_2/Conv2D/ReadVariableOpReadVariableOp2conv_model_conv_1_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02+
)conv_model/conv_1_2/Conv2D/ReadVariableOp�
conv_model/conv_1_2/Conv2DConv2D&conv_model/conv_1_1/Relu:activations:01conv_model/conv_1_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<< *
paddingVALID*
strides
2
conv_model/conv_1_2/Conv2D�
*conv_model/conv_1_2/BiasAdd/ReadVariableOpReadVariableOp3conv_model_conv_1_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv_model/conv_1_2/BiasAdd/ReadVariableOp�
conv_model/conv_1_2/BiasAddBiasAdd#conv_model/conv_1_2/Conv2D:output:02conv_model/conv_1_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<< 2
conv_model/conv_1_2/BiasAdd�
conv_model/conv_1_2/ReluRelu$conv_model/conv_1_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������<< 2
conv_model/conv_1_2/Relu�
conv_model/max_pool_1/MaxPoolMaxPool&conv_model/conv_1_2/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
conv_model/max_pool_1/MaxPool�
)conv_model/conv_2_1/Conv2D/ReadVariableOpReadVariableOp2conv_model_conv_2_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)conv_model/conv_2_1/Conv2D/ReadVariableOp�
conv_model/conv_2_1/Conv2DConv2D&conv_model/max_pool_1/MaxPool:output:01conv_model/conv_2_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv_model/conv_2_1/Conv2D�
*conv_model/conv_2_1/BiasAdd/ReadVariableOpReadVariableOp3conv_model_conv_2_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv_model/conv_2_1/BiasAdd/ReadVariableOp�
conv_model/conv_2_1/BiasAddBiasAdd#conv_model/conv_2_1/Conv2D:output:02conv_model/conv_2_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv_model/conv_2_1/BiasAdd�
conv_model/conv_2_1/ReluRelu$conv_model/conv_2_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv_model/conv_2_1/Relu�
)conv_model/conv_2_2/Conv2D/ReadVariableOpReadVariableOp2conv_model_conv_2_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)conv_model/conv_2_2/Conv2D/ReadVariableOp�
conv_model/conv_2_2/Conv2DConv2D&conv_model/conv_2_1/Relu:activations:01conv_model/conv_2_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv_model/conv_2_2/Conv2D�
*conv_model/conv_2_2/BiasAdd/ReadVariableOpReadVariableOp3conv_model_conv_2_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv_model/conv_2_2/BiasAdd/ReadVariableOp�
conv_model/conv_2_2/BiasAddBiasAdd#conv_model/conv_2_2/Conv2D:output:02conv_model/conv_2_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv_model/conv_2_2/BiasAdd�
conv_model/conv_2_2/ReluRelu$conv_model/conv_2_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv_model/conv_2_2/Relu�
conv_model/max_pool_2/MaxPoolMaxPool&conv_model/conv_2_2/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
conv_model/max_pool_2/MaxPool�
conv_model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@*  2
conv_model/flatten/Const�
conv_model/flatten/ReshapeReshape&conv_model/max_pool_2/MaxPool:output:0!conv_model/flatten/Const:output:0*
T0*(
_output_shapes
:����������T2
conv_model/flatten/Reshape�
&conv_model/logit/MatMul/ReadVariableOpReadVariableOp/conv_model_logit_matmul_readvariableop_resource*
_output_shapes
:	�T*
dtype02(
&conv_model/logit/MatMul/ReadVariableOp�
conv_model/logit/MatMulMatMul#conv_model/flatten/Reshape:output:0.conv_model/logit/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
conv_model/logit/MatMul�
'conv_model/logit/BiasAdd/ReadVariableOpReadVariableOp0conv_model_logit_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv_model/logit/BiasAdd/ReadVariableOp�
conv_model/logit/BiasAddBiasAdd!conv_model/logit/MatMul:product:0/conv_model/logit/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
conv_model/logit/BiasAdd�
conv_model/logit/SoftmaxSoftmax!conv_model/logit/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
conv_model/logit/Softmax�
IdentityIdentity"conv_model/logit/Softmax:softmax:0+^conv_model/conv_1_1/BiasAdd/ReadVariableOp*^conv_model/conv_1_1/Conv2D/ReadVariableOp+^conv_model/conv_1_2/BiasAdd/ReadVariableOp*^conv_model/conv_1_2/Conv2D/ReadVariableOp+^conv_model/conv_2_1/BiasAdd/ReadVariableOp*^conv_model/conv_2_1/Conv2D/ReadVariableOp+^conv_model/conv_2_2/BiasAdd/ReadVariableOp*^conv_model/conv_2_2/Conv2D/ReadVariableOp(^conv_model/logit/BiasAdd/ReadVariableOp'^conv_model/logit/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������@@::::::::::2X
*conv_model/conv_1_1/BiasAdd/ReadVariableOp*conv_model/conv_1_1/BiasAdd/ReadVariableOp2V
)conv_model/conv_1_1/Conv2D/ReadVariableOp)conv_model/conv_1_1/Conv2D/ReadVariableOp2X
*conv_model/conv_1_2/BiasAdd/ReadVariableOp*conv_model/conv_1_2/BiasAdd/ReadVariableOp2V
)conv_model/conv_1_2/Conv2D/ReadVariableOp)conv_model/conv_1_2/Conv2D/ReadVariableOp2X
*conv_model/conv_2_1/BiasAdd/ReadVariableOp*conv_model/conv_2_1/BiasAdd/ReadVariableOp2V
)conv_model/conv_2_1/Conv2D/ReadVariableOp)conv_model/conv_2_1/Conv2D/ReadVariableOp2X
*conv_model/conv_2_2/BiasAdd/ReadVariableOp*conv_model/conv_2_2/BiasAdd/ReadVariableOp2V
)conv_model/conv_2_2/Conv2D/ReadVariableOp)conv_model/conv_2_2/Conv2D/ReadVariableOp2R
'conv_model/logit/BiasAdd/ReadVariableOp'conv_model/logit/BiasAdd/ReadVariableOp2P
&conv_model/logit/MatMul/ReadVariableOp&conv_model/logit/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1
�
�
D__inference_conv_2_1_layer_call_and_return_conditional_losses_134106

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
)__inference_conv_2_2_layer_call_fn_134135

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv_2_2_layer_call_and_return_conditional_losses_1341272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
)__inference_conv_1_1_layer_call_fn_134060

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv_1_1_layer_call_and_return_conditional_losses_1340522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�E
�
"__inference__traced_restore_134410
file_prefix/
+assignvariableop_conv_model_conv_1_1_kernel/
+assignvariableop_1_conv_model_conv_1_1_bias1
-assignvariableop_2_conv_model_conv_1_2_kernel/
+assignvariableop_3_conv_model_conv_1_2_bias1
-assignvariableop_4_conv_model_conv_2_1_kernel/
+assignvariableop_5_conv_model_conv_2_1_bias1
-assignvariableop_6_conv_model_conv_2_2_kernel/
+assignvariableop_7_conv_model_conv_2_2_bias.
*assignvariableop_8_conv_model_logit_kernel,
(assignvariableop_9_conv_model_logit_bias 
assignvariableop_10_sgd_iter!
assignvariableop_11_sgd_decay)
%assignvariableop_12_sgd_learning_rate$
 assignvariableop_13_sgd_momentum
assignvariableop_14_total
assignvariableop_15_count
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B*conv_1_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB(conv_1_1/bias/.ATTRIBUTES/VARIABLE_VALUEB*conv_1_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB(conv_1_2/bias/.ATTRIBUTES/VARIABLE_VALUEB*conv_2_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB(conv_2_1/bias/.ATTRIBUTES/VARIABLE_VALUEB*conv_2_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB(conv_2_2/bias/.ATTRIBUTES/VARIABLE_VALUEB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp+assignvariableop_conv_model_conv_1_1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_conv_model_conv_1_1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp-assignvariableop_2_conv_model_conv_1_2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_conv_model_conv_1_2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp-assignvariableop_4_conv_model_conv_2_1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_conv_model_conv_2_1_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp-assignvariableop_6_conv_model_conv_2_2_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_conv_model_conv_2_2_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp*assignvariableop_8_conv_model_logit_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp(assignvariableop_9_conv_model_logit_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_sgd_iterIdentity_10:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_sgd_decayIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp%assignvariableop_12_sgd_learning_rateIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp assignvariableop_13_sgd_momentumIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16�
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�
�
)__inference_conv_2_1_layer_call_fn_134114

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv_2_1_layer_call_and_return_conditional_losses_1341062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
D__inference_conv_1_2_layer_call_and_return_conditional_losses_134073

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
G
+__inference_max_pool_1_layer_call_fn_134093

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_max_pool_1_layer_call_and_return_conditional_losses_1340872
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
b
F__inference_max_pool_1_layer_call_and_return_conditional_losses_134087

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
D
(__inference_flatten_layer_call_fn_134260

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������T*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1341772
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������T2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������@@<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
conv_1_1
conv_1_2

max_pool_1
conv_2_1
conv_2_2

max_pool_2
flatten
	dense
		optimizer

_training_endpoints
regularization_losses
trainable_variables
	variables
	keras_api

signatures
n__call__
o_default_save_signature
*p&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "ConvModel", "name": "conv_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "ConvModel"}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.0010000000474974513, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
q__call__
*r&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv_1_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 64, 64, 3], "config": {"name": "conv_1_1", "trainable": true, "batch_input_shape": [null, 64, 64, 3], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
s__call__
*t&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv_1_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_1_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�
regularization_losses
	variables
trainable_variables
	keras_api
u__call__
*v&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pool_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
w__call__
*x&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv_2_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_2_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
y__call__
*z&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv_2_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_2_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
,regularization_losses
-	variables
.trainable_variables
/	keras_api
{__call__
*|&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pool_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pool_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
0regularization_losses
1	variables
2trainable_variables
3	keras_api
}__call__
*~&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "logit", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "logit", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10816}}}}
I
:iter
	;decay
<learning_rate
=momentum"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
 4
!5
&6
'7
48
59"
trackable_list_wrapper
f
0
1
2
3
 4
!5
&6
'7
48
59"
trackable_list_wrapper
�
>non_trainable_variables

?layers
regularization_losses
@metrics
Alayer_regularization_losses
trainable_variables
	variables
n__call__
o_default_save_signature
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
4:2 2conv_model/conv_1_1/kernel
&:$ 2conv_model/conv_1_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Bnon_trainable_variables

Clayers
regularization_losses
Dmetrics
	variables
trainable_variables
Elayer_regularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
4:2  2conv_model/conv_1_2/kernel
&:$ 2conv_model/conv_1_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Fnon_trainable_variables

Glayers
regularization_losses
Hmetrics
	variables
trainable_variables
Ilayer_regularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jnon_trainable_variables

Klayers
regularization_losses
Lmetrics
	variables
trainable_variables
Mlayer_regularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
4:2 @2conv_model/conv_2_1/kernel
&:$@2conv_model/conv_2_1/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
�
Nnon_trainable_variables

Olayers
"regularization_losses
Pmetrics
#	variables
$trainable_variables
Qlayer_regularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
4:2@@2conv_model/conv_2_2/kernel
&:$@2conv_model/conv_2_2/bias
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
�
Rnon_trainable_variables

Slayers
(regularization_losses
Tmetrics
)	variables
*trainable_variables
Ulayer_regularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Vnon_trainable_variables

Wlayers
,regularization_losses
Xmetrics
-	variables
.trainable_variables
Ylayer_regularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Znon_trainable_variables

[layers
0regularization_losses
\metrics
1	variables
2trainable_variables
]layer_regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
*:(	�T2conv_model/logit/kernel
#:!2conv_model/logit/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
�
^non_trainable_variables

_layers
6regularization_losses
`metrics
7	variables
8trainable_variables
alayer_regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
'
b0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	ctotal
	dcount
e
_fn_kwargs
fregularization_losses
g	variables
htrainable_variables
i	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
jnon_trainable_variables

klayers
fregularization_losses
lmetrics
g	variables
htrainable_variables
mlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
+__inference_conv_model_layer_call_fn_134227�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������@@
�2�
!__inference__wrapped_model_134039�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������@@
�2�
F__inference_conv_model_layer_call_and_return_conditional_losses_134211�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������@@
�2�
)__inference_conv_1_1_layer_call_fn_134060�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
D__inference_conv_1_1_layer_call_and_return_conditional_losses_134052�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
)__inference_conv_1_2_layer_call_fn_134081�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
D__inference_conv_1_2_layer_call_and_return_conditional_losses_134073�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
+__inference_max_pool_1_layer_call_fn_134093�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
F__inference_max_pool_1_layer_call_and_return_conditional_losses_134087�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
)__inference_conv_2_1_layer_call_fn_134114�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
D__inference_conv_2_1_layer_call_and_return_conditional_losses_134106�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
)__inference_conv_2_2_layer_call_fn_134135�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
D__inference_conv_2_2_layer_call_and_return_conditional_losses_134127�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
+__inference_max_pool_2_layer_call_fn_134147�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
F__inference_max_pool_2_layer_call_and_return_conditional_losses_134141�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
(__inference_flatten_layer_call_fn_134260�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_flatten_layer_call_and_return_conditional_losses_134255�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_logit_layer_call_fn_134278�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_logit_layer_call_and_return_conditional_losses_134271�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
3B1
$__inference_signature_wrapper_134249input_1
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
!__inference__wrapped_model_134039{
 !&'458�5
.�+
)�&
input_1���������@@
� "3�0
.
output_1"�
output_1����������
D__inference_conv_1_1_layer_call_and_return_conditional_losses_134052�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
)__inference_conv_1_1_layer_call_fn_134060�I�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� �
D__inference_conv_1_2_layer_call_and_return_conditional_losses_134073�I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
)__inference_conv_1_2_layer_call_fn_134081�I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
D__inference_conv_2_1_layer_call_and_return_conditional_losses_134106� !I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������@
� �
)__inference_conv_2_1_layer_call_fn_134114� !I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+���������������������������@�
D__inference_conv_2_2_layer_call_and_return_conditional_losses_134127�&'I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
)__inference_conv_2_2_layer_call_fn_134135�&'I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
F__inference_conv_model_layer_call_and_return_conditional_losses_134211m
 !&'458�5
.�+
)�&
input_1���������@@
� "%�"
�
0���������
� �
+__inference_conv_model_layer_call_fn_134227`
 !&'458�5
.�+
)�&
input_1���������@@
� "�����������
C__inference_flatten_layer_call_and_return_conditional_losses_134255a7�4
-�*
(�%
inputs���������@
� "&�#
�
0����������T
� �
(__inference_flatten_layer_call_fn_134260T7�4
-�*
(�%
inputs���������@
� "�����������T�
A__inference_logit_layer_call_and_return_conditional_losses_134271]450�-
&�#
!�
inputs����������T
� "%�"
�
0���������
� z
&__inference_logit_layer_call_fn_134278P450�-
&�#
!�
inputs����������T
� "�����������
F__inference_max_pool_1_layer_call_and_return_conditional_losses_134087�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
+__inference_max_pool_1_layer_call_fn_134093�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
F__inference_max_pool_2_layer_call_and_return_conditional_losses_134141�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
+__inference_max_pool_2_layer_call_fn_134147�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
$__inference_signature_wrapper_134249�
 !&'45C�@
� 
9�6
4
input_1)�&
input_1���������@@"3�0
.
output_1"�
output_1���������