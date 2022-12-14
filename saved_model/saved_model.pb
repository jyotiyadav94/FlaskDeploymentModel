??'
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-0-g919f693420e8??"
?
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
?
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
?
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
?
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
?
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameblock2_conv1/kernel
?
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@?*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:?*
dtype0
?
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock2_conv2/kernel
?
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv1/kernel
?
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:?*
dtype0
?
block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv2/kernel
?
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv3/kernel
?
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:?*
dtype0
?
block3_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv4/kernel
?
'block3_conv4/kernel/Read/ReadVariableOpReadVariableOpblock3_conv4/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv4/bias
t
%block3_conv4/bias/Read/ReadVariableOpReadVariableOpblock3_conv4/bias*
_output_shapes	
:?*
dtype0
?
block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv1/kernel
?
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:?*
dtype0
?
block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv2/kernel
?
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:?*
dtype0
?
block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv3/kernel
?
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:?*
dtype0
?
block4_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv4/kernel
?
'block4_conv4/kernel/Read/ReadVariableOpReadVariableOpblock4_conv4/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv4/bias
t
%block4_conv4/bias/Read/ReadVariableOpReadVariableOpblock4_conv4/bias*
_output_shapes	
:?*
dtype0
?
block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv1/kernel
?
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:?*
dtype0
?
block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv2/kernel
?
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:?*
dtype0
?
block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv3/kernel
?
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:?*
dtype0
?
block5_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv4/kernel
?
'block5_conv4/kernel/Read/ReadVariableOpReadVariableOpblock5_conv4/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv4/bias
t
%block5_conv4/bias/Read/ReadVariableOpReadVariableOpblock5_conv4/bias*
_output_shapes	
:?*
dtype0
?
block_6_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*&
shared_nameblock_6_conv_1/kernel
?
)block_6_conv_1/kernel/Read/ReadVariableOpReadVariableOpblock_6_conv_1/kernel*(
_output_shapes
:??*
dtype0

block_6_conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameblock_6_conv_1/bias
x
'block_6_conv_1/bias/Read/ReadVariableOpReadVariableOpblock_6_conv_1/bias*
_output_shapes	
:?*
dtype0
?
block_6_conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*&
shared_nameblock_6_conv_2/kernel
?
)block_6_conv_2/kernel/Read/ReadVariableOpReadVariableOpblock_6_conv_2/kernel*(
_output_shapes
:??*
dtype0

block_6_conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameblock_6_conv_2/bias
x
'block_6_conv_2/bias/Read/ReadVariableOpReadVariableOpblock_6_conv_2/bias*
_output_shapes	
:?*
dtype0
?
col_decoder/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_namecol_decoder/conv2d/kernel
?
-col_decoder/conv2d/kernel/Read/ReadVariableOpReadVariableOpcol_decoder/conv2d/kernel*(
_output_shapes
:??*
dtype0
?
col_decoder/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namecol_decoder/conv2d/bias
?
+col_decoder/conv2d/bias/Read/ReadVariableOpReadVariableOpcol_decoder/conv2d/bias*
_output_shapes	
:?*
dtype0
?
#col_decoder/conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#col_decoder/conv2d_transpose/kernel
?
7col_decoder/conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOp#col_decoder/conv2d_transpose/kernel*'
_output_shapes
:?*
dtype0
?
!col_decoder/conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!col_decoder/conv2d_transpose/bias
?
5col_decoder/conv2d_transpose/bias/Read/ReadVariableOpReadVariableOp!col_decoder/conv2d_transpose/bias*
_output_shapes
:*
dtype0
?
table_decoder/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*.
shared_nametable_decoder/conv2d_1/kernel
?
1table_decoder/conv2d_1/kernel/Read/ReadVariableOpReadVariableOptable_decoder/conv2d_1/kernel*(
_output_shapes
:??*
dtype0
?
table_decoder/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nametable_decoder/conv2d_1/bias
?
/table_decoder/conv2d_1/bias/Read/ReadVariableOpReadVariableOptable_decoder/conv2d_1/bias*
_output_shapes	
:?*
dtype0
?
table_decoder/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*.
shared_nametable_decoder/conv2d_2/kernel
?
1table_decoder/conv2d_2/kernel/Read/ReadVariableOpReadVariableOptable_decoder/conv2d_2/kernel*(
_output_shapes
:??*
dtype0
?
table_decoder/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nametable_decoder/conv2d_2/bias
?
/table_decoder/conv2d_2/bias/Read/ReadVariableOpReadVariableOptable_decoder/conv2d_2/bias*
_output_shapes	
:?*
dtype0
?
'table_decoder/conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'table_decoder/conv2d_transpose_1/kernel
?
;table_decoder/conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp'table_decoder/conv2d_transpose_1/kernel*'
_output_shapes
:?*
dtype0
?
%table_decoder/conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%table_decoder/conv2d_transpose_1/bias
?
9table_decoder/conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOp%table_decoder/conv2d_transpose_1/bias*
_output_shapes
:*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer_with_weights-15
layer-20
layer-21
layer_with_weights-16
layer-22
layer-23
layer_with_weights-17
layer-24
layer-25
layer_with_weights-18
layer-26
layer_with_weights-19
layer-27
	optimizer
loss
	variables
 trainable_variables
!regularization_losses
"	keras_api
#
signatures
 
h

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
h

*kernel
+bias
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
h

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
R
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
h

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
R
\regularization_losses
]	variables
^trainable_variables
_	keras_api
h

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
h

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
h

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
h

rkernel
sbias
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
R
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
j

|kernel
}bias
~regularization_losses
	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?

?conv1
?	upsample1
?	upsample2
?	upsample3
?	upsample4
?convtraspose
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?

?conv1

?conv2

?drop1
?	upsample1
?	upsample2
?	upsample3
?	upsample4
?convtraspose
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
 
?
$0
%1
*2
+3
44
55
:6
;7
D8
E9
J10
K11
P12
Q13
V14
W15
`16
a17
f18
g19
l20
m21
r22
s23
|24
}25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
t
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
 
?
?non_trainable_variables
?metrics
	variables
 ?layer_regularization_losses
?layers
 trainable_variables
?layer_metrics
!regularization_losses
 
_]
VARIABLE_VALUEblock1_conv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1
 
?
&regularization_losses
?metrics
'	variables
 ?layer_regularization_losses
?layers
(trainable_variables
?layer_metrics
?non_trainable_variables
_]
VARIABLE_VALUEblock1_conv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1
 
?
,regularization_losses
?metrics
-	variables
 ?layer_regularization_losses
?layers
.trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
0regularization_losses
?metrics
1	variables
 ?layer_regularization_losses
?layers
2trainable_variables
?layer_metrics
?non_trainable_variables
_]
VARIABLE_VALUEblock2_conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51
 
?
6regularization_losses
?metrics
7	variables
 ?layer_regularization_losses
?layers
8trainable_variables
?layer_metrics
?non_trainable_variables
_]
VARIABLE_VALUEblock2_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1
 
?
<regularization_losses
?metrics
=	variables
 ?layer_regularization_losses
?layers
>trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
@regularization_losses
?metrics
A	variables
 ?layer_regularization_losses
?layers
Btrainable_variables
?layer_metrics
?non_trainable_variables
_]
VARIABLE_VALUEblock3_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1
 
?
Fregularization_losses
?metrics
G	variables
 ?layer_regularization_losses
?layers
Htrainable_variables
?layer_metrics
?non_trainable_variables
_]
VARIABLE_VALUEblock3_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1
 
?
Lregularization_losses
?metrics
M	variables
 ?layer_regularization_losses
?layers
Ntrainable_variables
?layer_metrics
?non_trainable_variables
_]
VARIABLE_VALUEblock3_conv3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1
 
?
Rregularization_losses
?metrics
S	variables
 ?layer_regularization_losses
?layers
Ttrainable_variables
?layer_metrics
?non_trainable_variables
_]
VARIABLE_VALUEblock3_conv4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1
 
?
Xregularization_losses
?metrics
Y	variables
 ?layer_regularization_losses
?layers
Ztrainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
\regularization_losses
?metrics
]	variables
 ?layer_regularization_losses
?layers
^trainable_variables
?layer_metrics
?non_trainable_variables
_]
VARIABLE_VALUEblock4_conv1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

`0
a1
 
?
bregularization_losses
?metrics
c	variables
 ?layer_regularization_losses
?layers
dtrainable_variables
?layer_metrics
?non_trainable_variables
_]
VARIABLE_VALUEblock4_conv2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1
 
?
hregularization_losses
?metrics
i	variables
 ?layer_regularization_losses
?layers
jtrainable_variables
?layer_metrics
?non_trainable_variables
`^
VARIABLE_VALUEblock4_conv3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock4_conv3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1
 
?
nregularization_losses
?metrics
o	variables
 ?layer_regularization_losses
?layers
ptrainable_variables
?layer_metrics
?non_trainable_variables
`^
VARIABLE_VALUEblock4_conv4/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock4_conv4/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

r0
s1
 
?
tregularization_losses
?metrics
u	variables
 ?layer_regularization_losses
?layers
vtrainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
xregularization_losses
?metrics
y	variables
 ?layer_regularization_losses
?layers
ztrainable_variables
?layer_metrics
?non_trainable_variables
`^
VARIABLE_VALUEblock5_conv1/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv1/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

|0
}1
 
?
~regularization_losses
?metrics
	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
`^
VARIABLE_VALUEblock5_conv2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
`^
VARIABLE_VALUEblock5_conv3/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv3/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
`^
VARIABLE_VALUEblock5_conv4/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv4/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
b`
VARIABLE_VALUEblock_6_conv_1/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEblock_6_conv_1/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
b`
VARIABLE_VALUEblock_6_conv_2/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEblock_6_conv_2/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
 
?0
?1
?2
?3
 
?0
?1
?2
?3
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
0
?0
?1
?2
?3
?4
?5
0
?0
?1
?2
?3
?4
?5
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
VT
VARIABLE_VALUEcol_decoder/conv2d/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcol_decoder/conv2d/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#col_decoder/conv2d_transpose/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!col_decoder/conv2d_transpose/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtable_decoder/conv2d_1/kernel'variables/40/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEtable_decoder/conv2d_1/bias'variables/41/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtable_decoder/conv2d_2/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEtable_decoder/conv2d_2/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'table_decoder/conv2d_transpose_1/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%table_decoder/conv2d_transpose_1/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE
?
$0
%1
*2
+3
44
55
:6
;7
D8
E9
J10
K11
P12
Q13
V14
W15
`16
a17
f18
g19
l20
m21
r22
s23
|24
}25
?26
?27
?28
?29
?30
?31

?0
?1
?2
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
 
 
 
 
 

$0
%1
 
 
 
 

*0
+1
 
 
 
 
 
 
 
 
 

40
51
 
 
 
 

:0
;1
 
 
 
 
 
 
 
 
 

D0
E1
 
 
 
 

J0
K1
 
 
 
 

P0
Q1
 
 
 
 

V0
W1
 
 
 
 
 
 
 
 
 

`0
a1
 
 
 
 

f0
g1
 
 
 
 

l0
m1
 
 
 
 

r0
s1
 
 
 
 
 
 
 
 
 

|0
}1
 
 
 
 

?0
?1
 
 
 
 

?0
?1
 
 
 
 

?0
?1
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

?0
?1

?0
?1
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 

?0
?1

?0
?1
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
0
?0
?1
?2
?3
?4
?5
 
 
 

?0
?1

?0
?1
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 

?0
?1

?0
?1
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 

?0
?1

?0
?1
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
 
 
@
?0
?1
?2
?3
?4
?5
?6
?7
 
 
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
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
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
?
serving_default_Input_LayerPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_Input_Layerblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasblock_6_conv_1/kernelblock_6_conv_1/biasblock_6_conv_2/kernelblock_6_conv_2/biastable_decoder/conv2d_1/kerneltable_decoder/conv2d_1/biastable_decoder/conv2d_2/kerneltable_decoder/conv2d_2/bias'table_decoder/conv2d_transpose_1/kernel%table_decoder/conv2d_transpose_1/biascol_decoder/conv2d/kernelcol_decoder/conv2d/bias#col_decoder/conv2d_transpose/kernel!col_decoder/conv2d_transpose/bias*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::???????????:???????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_3490
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block3_conv4/kernel/Read/ReadVariableOp%block3_conv4/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block4_conv4/kernel/Read/ReadVariableOp%block4_conv4/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp'block5_conv4/kernel/Read/ReadVariableOp%block5_conv4/bias/Read/ReadVariableOp)block_6_conv_1/kernel/Read/ReadVariableOp'block_6_conv_1/bias/Read/ReadVariableOp)block_6_conv_2/kernel/Read/ReadVariableOp'block_6_conv_2/bias/Read/ReadVariableOp-col_decoder/conv2d/kernel/Read/ReadVariableOp+col_decoder/conv2d/bias/Read/ReadVariableOp7col_decoder/conv2d_transpose/kernel/Read/ReadVariableOp5col_decoder/conv2d_transpose/bias/Read/ReadVariableOp1table_decoder/conv2d_1/kernel/Read/ReadVariableOp/table_decoder/conv2d_1/bias/Read/ReadVariableOp1table_decoder/conv2d_2/kernel/Read/ReadVariableOp/table_decoder/conv2d_2/bias/Read/ReadVariableOp;table_decoder/conv2d_transpose_1/kernel/Read/ReadVariableOp9table_decoder/conv2d_transpose_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpConst*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference__traced_save_5340
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasblock_6_conv_1/kernelblock_6_conv_1/biasblock_6_conv_2/kernelblock_6_conv_2/biascol_decoder/conv2d/kernelcol_decoder/conv2d/bias#col_decoder/conv2d_transpose/kernel!col_decoder/conv2d_transpose/biastable_decoder/conv2d_1/kerneltable_decoder/conv2d_1/biastable_decoder/conv2d_2/kerneltable_decoder/conv2d_2/bias'table_decoder/conv2d_transpose_1/kernel%table_decoder/conv2d_transpose_1/biastotalcounttotal_1count_1total_2count_2*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_restore_5506??
?
?
+__inference_block3_conv4_layer_call_fn_4389

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_19012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
+__inference_block1_conv1_layer_call_fn_4209

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_17702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
+__inference_block4_conv4_layer_call_fn_4489

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_19752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
F
*__inference_block2_pool_layer_call_fn_4309

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_18372
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_4294

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_4646

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?&
?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_5040

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SoftmaxSoftmaxBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block2_conv1_layer_call_and_return_conditional_losses_1810

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
1__inference_conv2d_transpose_1_layer_call_fn_5160

inputs"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_17022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_block2_pool_layer_call_fn_4304

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_12092
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_4394

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_block1_conv2_layer_call_fn_4229

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_17872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
F__inference_block3_conv4_layer_call_and_return_conditional_losses_1901

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv4_layer_call_and_return_conditional_losses_2049

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_1985

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????22?*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????22?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????dd?:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
?
F__inference_block3_conv3_layer_call_and_return_conditional_losses_4360

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv4_layer_call_and_return_conditional_losses_4580

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
??
?"
 __inference__traced_restore_5506
file_prefix>
$assignvariableop_block1_conv1_kernel:@2
$assignvariableop_1_block1_conv1_bias:@@
&assignvariableop_2_block1_conv2_kernel:@@2
$assignvariableop_3_block1_conv2_bias:@A
&assignvariableop_4_block2_conv1_kernel:@?3
$assignvariableop_5_block2_conv1_bias:	?B
&assignvariableop_6_block2_conv2_kernel:??3
$assignvariableop_7_block2_conv2_bias:	?B
&assignvariableop_8_block3_conv1_kernel:??3
$assignvariableop_9_block3_conv1_bias:	?C
'assignvariableop_10_block3_conv2_kernel:??4
%assignvariableop_11_block3_conv2_bias:	?C
'assignvariableop_12_block3_conv3_kernel:??4
%assignvariableop_13_block3_conv3_bias:	?C
'assignvariableop_14_block3_conv4_kernel:??4
%assignvariableop_15_block3_conv4_bias:	?C
'assignvariableop_16_block4_conv1_kernel:??4
%assignvariableop_17_block4_conv1_bias:	?C
'assignvariableop_18_block4_conv2_kernel:??4
%assignvariableop_19_block4_conv2_bias:	?C
'assignvariableop_20_block4_conv3_kernel:??4
%assignvariableop_21_block4_conv3_bias:	?C
'assignvariableop_22_block4_conv4_kernel:??4
%assignvariableop_23_block4_conv4_bias:	?C
'assignvariableop_24_block5_conv1_kernel:??4
%assignvariableop_25_block5_conv1_bias:	?C
'assignvariableop_26_block5_conv2_kernel:??4
%assignvariableop_27_block5_conv2_bias:	?C
'assignvariableop_28_block5_conv3_kernel:??4
%assignvariableop_29_block5_conv3_bias:	?C
'assignvariableop_30_block5_conv4_kernel:??4
%assignvariableop_31_block5_conv4_bias:	?E
)assignvariableop_32_block_6_conv_1_kernel:??6
'assignvariableop_33_block_6_conv_1_bias:	?E
)assignvariableop_34_block_6_conv_2_kernel:??6
'assignvariableop_35_block_6_conv_2_bias:	?I
-assignvariableop_36_col_decoder_conv2d_kernel:??:
+assignvariableop_37_col_decoder_conv2d_bias:	?R
7assignvariableop_38_col_decoder_conv2d_transpose_kernel:?C
5assignvariableop_39_col_decoder_conv2d_transpose_bias:M
1assignvariableop_40_table_decoder_conv2d_1_kernel:??>
/assignvariableop_41_table_decoder_conv2d_1_bias:	?M
1assignvariableop_42_table_decoder_conv2d_2_kernel:??>
/assignvariableop_43_table_decoder_conv2d_2_bias:	?V
;assignvariableop_44_table_decoder_conv2d_transpose_1_kernel:?G
9assignvariableop_45_table_decoder_conv2d_transpose_1_bias:#
assignvariableop_46_total: #
assignvariableop_47_count: %
assignvariableop_48_total_1: %
assignvariableop_49_count_1: %
assignvariableop_50_total_2: %
assignvariableop_51_count_2: 
identity_53??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*?
value?B?5B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
7252
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp$assignvariableop_block1_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_block1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_block1_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_block1_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block2_conv1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block2_conv1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block2_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block2_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block3_conv1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block3_conv1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block3_conv2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block3_conv2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block3_conv3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block3_conv3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block3_conv4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block3_conv4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block4_conv1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block4_conv1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block4_conv2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block4_conv2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block4_conv3_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block4_conv3_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block4_conv4_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block4_conv4_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block5_conv1_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block5_conv1_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_block5_conv2_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_block5_conv2_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_block5_conv3_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp%assignvariableop_29_block5_conv3_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_block5_conv4_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp%assignvariableop_31_block5_conv4_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_block_6_conv_1_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_block_6_conv_1_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_block_6_conv_2_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_block_6_conv_2_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp-assignvariableop_36_col_decoder_conv2d_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_col_decoder_conv2d_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp7assignvariableop_38_col_decoder_conv2d_transpose_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp5assignvariableop_39_col_decoder_conv2d_transpose_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp1assignvariableop_40_table_decoder_conv2d_1_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp/assignvariableop_41_table_decoder_conv2d_1_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp1assignvariableop_42_table_decoder_conv2d_2_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp/assignvariableop_43_table_decoder_conv2d_2_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp;assignvariableop_44_table_decoder_conv2d_transpose_1_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp9assignvariableop_45_table_decoder_conv2d_transpose_1_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_totalIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_countIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_total_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_count_1Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_total_2Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_count_2Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_519
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_52f
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_53?	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_53Identity_53:output:0*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4950

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_2346
input_layer!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?&

unknown_33:??

unknown_34:	?&

unknown_35:??

unknown_36:	?&

unknown_37:??

unknown_38:	?%

unknown_39:?

unknown_40:&

unknown_41:??

unknown_42:	?%

unknown_43:?

unknown_44:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::???????????:???????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_22492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameInput_Layer
?&
?
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_5151

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SoftmaxSoftmaxBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_2249

inputs+
block1_conv1_1771:@
block1_conv1_1773:@+
block1_conv2_1788:@@
block1_conv2_1790:@,
block2_conv1_1811:@? 
block2_conv1_1813:	?-
block2_conv2_1828:?? 
block2_conv2_1830:	?-
block3_conv1_1851:?? 
block3_conv1_1853:	?-
block3_conv2_1868:?? 
block3_conv2_1870:	?-
block3_conv3_1885:?? 
block3_conv3_1887:	?-
block3_conv4_1902:?? 
block3_conv4_1904:	?-
block4_conv1_1925:?? 
block4_conv1_1927:	?-
block4_conv2_1942:?? 
block4_conv2_1944:	?-
block4_conv3_1959:?? 
block4_conv3_1961:	?-
block4_conv4_1976:?? 
block4_conv4_1978:	?-
block5_conv1_1999:?? 
block5_conv1_2001:	?-
block5_conv2_2016:?? 
block5_conv2_2018:	?-
block5_conv3_2033:?? 
block5_conv3_2035:	?-
block5_conv4_2050:?? 
block5_conv4_2052:	?/
block_6_conv_1_2073:??"
block_6_conv_1_2075:	?/
block_6_conv_2_2097:??"
block_6_conv_2_2099:	?.
table_decoder_2171:??!
table_decoder_2173:	?.
table_decoder_2175:??!
table_decoder_2177:	?-
table_decoder_2179:? 
table_decoder_2181:,
col_decoder_2238:??
col_decoder_2240:	?+
col_decoder_2242:?
col_decoder_2244:
identity

identity_1??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block3_conv4/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block4_conv4/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?$block5_conv4/StatefulPartitionedCall?&block_6_conv_1/StatefulPartitionedCall?&block_6_conv_2/StatefulPartitionedCall?#col_decoder/StatefulPartitionedCall?%table_decoder/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_1771block1_conv1_1773*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_17702&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_1788block1_conv2_1790*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_17872&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_17972
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_1811block2_conv1_1813*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_18102&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_1828block2_conv2_1830*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_18272&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_18372
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_1851block3_conv1_1853*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_18502&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_1868block3_conv2_1870*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_18672&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_1885block3_conv3_1887*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_18842&
$block3_conv3/StatefulPartitionedCall?
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_1902block3_conv4_1904*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_19012&
$block3_conv4/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_19112
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_1925block4_conv1_1927*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_19242&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_1942block4_conv2_1944*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_19412&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_1959block4_conv3_1961*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_19582&
$block4_conv3/StatefulPartitionedCall?
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_1976block4_conv4_1978*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_19752&
$block4_conv4/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_19852
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_1999block5_conv1_2001*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_19982&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_2016block5_conv2_2018*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_20152&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_2033block5_conv3_2035*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_20322&
$block5_conv3/StatefulPartitionedCall?
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_2050block5_conv4_2052*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_20492&
$block5_conv4/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_20592
block5_pool/PartitionedCall?
&block_6_conv_1/StatefulPartitionedCallStatefulPartitionedCall$block5_pool/PartitionedCall:output:0block_6_conv_1_2073block_6_conv_1_2075*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_20722(
&block_6_conv_1/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall/block_6_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_20832
dropout/PartitionedCall?
&block_6_conv_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0block_6_conv_2_2097block_6_conv_2_2099*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_20962(
&block_6_conv_2/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall/block_6_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_21072
dropout_1/PartitionedCall?
%table_decoder/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0table_decoder_2171table_decoder_2173table_decoder_2175table_decoder_2177table_decoder_2179table_decoder_2181*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_21702'
%table_decoder/StatefulPartitionedCall?
#col_decoder/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0col_decoder_2238col_decoder_2240col_decoder_2242col_decoder_2244*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_col_decoder_layer_call_and_return_conditional_losses_22372%
#col_decoder/StatefulPartitionedCall?
IdentityIdentity,col_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity.table_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2P
&block_6_conv_1/StatefulPartitionedCall&block_6_conv_1/StatefulPartitionedCall2P
&block_6_conv_2/StatefulPartitionedCall&block_6_conv_2/StatefulPartitionedCall2J
#col_decoder/StatefulPartitionedCall#col_decoder/StatefulPartitionedCall2N
%table_decoder/StatefulPartitionedCall%table_decoder/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?(
?__inference_model_layer_call_and_return_conditional_losses_3730

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@?;
,block2_conv1_biasadd_readvariableop_resource:	?G
+block2_conv2_conv2d_readvariableop_resource:??;
,block2_conv2_biasadd_readvariableop_resource:	?G
+block3_conv1_conv2d_readvariableop_resource:??;
,block3_conv1_biasadd_readvariableop_resource:	?G
+block3_conv2_conv2d_readvariableop_resource:??;
,block3_conv2_biasadd_readvariableop_resource:	?G
+block3_conv3_conv2d_readvariableop_resource:??;
,block3_conv3_biasadd_readvariableop_resource:	?G
+block3_conv4_conv2d_readvariableop_resource:??;
,block3_conv4_biasadd_readvariableop_resource:	?G
+block4_conv1_conv2d_readvariableop_resource:??;
,block4_conv1_biasadd_readvariableop_resource:	?G
+block4_conv2_conv2d_readvariableop_resource:??;
,block4_conv2_biasadd_readvariableop_resource:	?G
+block4_conv3_conv2d_readvariableop_resource:??;
,block4_conv3_biasadd_readvariableop_resource:	?G
+block4_conv4_conv2d_readvariableop_resource:??;
,block4_conv4_biasadd_readvariableop_resource:	?G
+block5_conv1_conv2d_readvariableop_resource:??;
,block5_conv1_biasadd_readvariableop_resource:	?G
+block5_conv2_conv2d_readvariableop_resource:??;
,block5_conv2_biasadd_readvariableop_resource:	?G
+block5_conv3_conv2d_readvariableop_resource:??;
,block5_conv3_biasadd_readvariableop_resource:	?G
+block5_conv4_conv2d_readvariableop_resource:??;
,block5_conv4_biasadd_readvariableop_resource:	?I
-block_6_conv_1_conv2d_readvariableop_resource:??=
.block_6_conv_1_biasadd_readvariableop_resource:	?I
-block_6_conv_2_conv2d_readvariableop_resource:??=
.block_6_conv_2_biasadd_readvariableop_resource:	?Q
5table_decoder_conv2d_1_conv2d_readvariableop_resource:??E
6table_decoder_conv2d_1_biasadd_readvariableop_resource:	?Q
5table_decoder_conv2d_2_conv2d_readvariableop_resource:??E
6table_decoder_conv2d_2_biasadd_readvariableop_resource:	?d
Itable_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:?N
@table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource:M
1col_decoder_conv2d_conv2d_readvariableop_resource:??A
2col_decoder_conv2d_biasadd_readvariableop_resource:	?`
Ecol_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource:?J
<col_decoder_conv2d_transpose_biasadd_readvariableop_resource:
identity

identity_1??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block3_conv4/BiasAdd/ReadVariableOp?"block3_conv4/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block4_conv4/BiasAdd/ReadVariableOp?"block4_conv4/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?#block5_conv4/BiasAdd/ReadVariableOp?"block5_conv4/Conv2D/ReadVariableOp?%block_6_conv_1/BiasAdd/ReadVariableOp?$block_6_conv_1/Conv2D/ReadVariableOp?%block_6_conv_2/BiasAdd/ReadVariableOp?$block_6_conv_2/Conv2D/ReadVariableOp?)col_decoder/conv2d/BiasAdd/ReadVariableOp?(col_decoder/conv2d/Conv2D/ReadVariableOp?3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp?<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?-table_decoder/conv2d_1/BiasAdd/ReadVariableOp?,table_decoder/conv2d_1/Conv2D/ReadVariableOp?-table_decoder/conv2d_2/BiasAdd/ReadVariableOp?,table_decoder/conv2d_2/Conv2D/ReadVariableOp?7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/Relu?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*2
_output_shapes 
:????????????*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block3_conv1/Relu?
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block3_conv2/Conv2D?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block3_conv2/BiasAdd?
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block3_conv2/Relu?
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block3_conv3/Conv2D?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block3_conv3/BiasAdd?
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block3_conv3/Relu?
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv4/Conv2D/ReadVariableOp?
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block3_conv4/Conv2D?
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv4/BiasAdd/ReadVariableOp?
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block3_conv4/BiasAdd?
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block3_conv4/Relu?
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:?????????dd?*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
block4_conv1/Conv2D?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv1/BiasAdd?
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv1/Relu?
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
block4_conv2/Conv2D?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv2/BiasAdd?
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv2/Relu?
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
block4_conv3/Conv2D?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv3/BiasAdd?
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv3/Relu?
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv4/Conv2D/ReadVariableOp?
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
block4_conv4/Conv2D?
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv4/BiasAdd/ReadVariableOp?
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv4/BiasAdd?
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv4/Relu?
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:?????????22?*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
block5_conv1/Conv2D?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
block5_conv1/BiasAdd?
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
block5_conv1/Relu?
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv2/Conv2D/ReadVariableOp?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
block5_conv2/Conv2D?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
block5_conv2/BiasAdd?
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
block5_conv2/Relu?
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv3/Conv2D/ReadVariableOp?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
block5_conv3/Conv2D?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
block5_conv3/BiasAdd?
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
block5_conv3/Relu?
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv4/Conv2D/ReadVariableOp?
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
block5_conv4/Conv2D?
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv4/BiasAdd/ReadVariableOp?
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
block5_conv4/BiasAdd?
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
block5_conv4/Relu?
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool?
$block_6_conv_1/Conv2D/ReadVariableOpReadVariableOp-block_6_conv_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$block_6_conv_1/Conv2D/ReadVariableOp?
block_6_conv_1/Conv2DConv2Dblock5_pool/MaxPool:output:0,block_6_conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
block_6_conv_1/Conv2D?
%block_6_conv_1/BiasAdd/ReadVariableOpReadVariableOp.block_6_conv_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%block_6_conv_1/BiasAdd/ReadVariableOp?
block_6_conv_1/BiasAddBiasAddblock_6_conv_1/Conv2D:output:0-block_6_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block_6_conv_1/BiasAdd?
block_6_conv_1/ReluRelublock_6_conv_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block_6_conv_1/Relu?
dropout/IdentityIdentity!block_6_conv_1/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout/Identity?
$block_6_conv_2/Conv2D/ReadVariableOpReadVariableOp-block_6_conv_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$block_6_conv_2/Conv2D/ReadVariableOp?
block_6_conv_2/Conv2DConv2Ddropout/Identity:output:0,block_6_conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
block_6_conv_2/Conv2D?
%block_6_conv_2/BiasAdd/ReadVariableOpReadVariableOp.block_6_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%block_6_conv_2/BiasAdd/ReadVariableOp?
block_6_conv_2/BiasAddBiasAddblock_6_conv_2/Conv2D:output:0-block_6_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block_6_conv_2/BiasAdd?
block_6_conv_2/ReluRelublock_6_conv_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block_6_conv_2/Relu?
dropout_1/IdentityIdentity!block_6_conv_2/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_1/Identity?
,table_decoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5table_decoder_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02.
,table_decoder/conv2d_1/Conv2D/ReadVariableOp?
table_decoder/conv2d_1/Conv2DConv2Ddropout_1/Identity:output:04table_decoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
table_decoder/conv2d_1/Conv2D?
-table_decoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6table_decoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-table_decoder/conv2d_1/BiasAdd/ReadVariableOp?
table_decoder/conv2d_1/BiasAddBiasAdd&table_decoder/conv2d_1/Conv2D:output:05table_decoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
table_decoder/conv2d_1/BiasAdd?
table_decoder/conv2d_1/ReluRelu'table_decoder/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
table_decoder/conv2d_1/Relu?
 table_decoder/dropout_2/IdentityIdentity)table_decoder/conv2d_1/Relu:activations:0*
T0*0
_output_shapes
:??????????2"
 table_decoder/dropout_2/Identity?
,table_decoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5table_decoder_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02.
,table_decoder/conv2d_2/Conv2D/ReadVariableOp?
table_decoder/conv2d_2/Conv2DConv2D)table_decoder/dropout_2/Identity:output:04table_decoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
table_decoder/conv2d_2/Conv2D?
-table_decoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6table_decoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-table_decoder/conv2d_2/BiasAdd/ReadVariableOp?
table_decoder/conv2d_2/BiasAddBiasAdd&table_decoder/conv2d_2/Conv2D:output:05table_decoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
table_decoder/conv2d_2/BiasAdd?
table_decoder/conv2d_2/ReluRelu'table_decoder/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
table_decoder/conv2d_2/Relu?
#table_decoder/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#table_decoder/up_sampling2d_4/Const?
%table_decoder/up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2'
%table_decoder/up_sampling2d_4/Const_1?
!table_decoder/up_sampling2d_4/mulMul,table_decoder/up_sampling2d_4/Const:output:0.table_decoder/up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_4/mul?
3table_decoder/up_sampling2d_4/resize/ResizeBilinearResizeBilinear)table_decoder/conv2d_1/Relu:activations:0%table_decoder/up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(25
3table_decoder/up_sampling2d_4/resize/ResizeBilinear?
%table_decoder/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%table_decoder/concatenate/concat/axis?
 table_decoder/concatenate/concatConcatV2block4_pool/MaxPool:output:0Dtable_decoder/up_sampling2d_4/resize/ResizeBilinear:resized_images:0.table_decoder/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2"
 table_decoder/concatenate/concat?
#table_decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2%
#table_decoder/up_sampling2d_5/Const?
%table_decoder/up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2'
%table_decoder/up_sampling2d_5/Const_1?
!table_decoder/up_sampling2d_5/mulMul,table_decoder/up_sampling2d_5/Const:output:0.table_decoder/up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_5/mul?
3table_decoder/up_sampling2d_5/resize/ResizeBilinearResizeBilinear)table_decoder/concatenate/concat:output:0%table_decoder/up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(25
3table_decoder/up_sampling2d_5/resize/ResizeBilinear?
'table_decoder/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'table_decoder/concatenate_1/concat/axis?
"table_decoder/concatenate_1/concatConcatV2block3_pool/MaxPool:output:0Dtable_decoder/up_sampling2d_5/resize/ResizeBilinear:resized_images:00table_decoder/concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2$
"table_decoder/concatenate_1/concat?
#table_decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2%
#table_decoder/up_sampling2d_6/Const?
%table_decoder/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2'
%table_decoder/up_sampling2d_6/Const_1?
!table_decoder/up_sampling2d_6/mulMul,table_decoder/up_sampling2d_6/Const:output:0.table_decoder/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_6/mul?
:table_decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor+table_decoder/concatenate_1/concat:output:0%table_decoder/up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2<
:table_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor?
#table_decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2%
#table_decoder/up_sampling2d_7/Const?
%table_decoder/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2'
%table_decoder/up_sampling2d_7/Const_1?
!table_decoder/up_sampling2d_7/mulMul,table_decoder/up_sampling2d_7/Const:output:0.table_decoder/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_7/mul?
:table_decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborKtable_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0%table_decoder/up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2<
:table_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor?
&table_decoder/conv2d_transpose_1/ShapeShapeKtable_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2(
&table_decoder/conv2d_transpose_1/Shape?
4table_decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4table_decoder/conv2d_transpose_1/strided_slice/stack?
6table_decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice/stack_1?
6table_decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice/stack_2?
.table_decoder/conv2d_transpose_1/strided_sliceStridedSlice/table_decoder/conv2d_transpose_1/Shape:output:0=table_decoder/conv2d_transpose_1/strided_slice/stack:output:0?table_decoder/conv2d_transpose_1/strided_slice/stack_1:output:0?table_decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.table_decoder/conv2d_transpose_1/strided_slice?
(table_decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2*
(table_decoder/conv2d_transpose_1/stack/1?
(table_decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2*
(table_decoder/conv2d_transpose_1/stack/2?
(table_decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(table_decoder/conv2d_transpose_1/stack/3?
&table_decoder/conv2d_transpose_1/stackPack7table_decoder/conv2d_transpose_1/strided_slice:output:01table_decoder/conv2d_transpose_1/stack/1:output:01table_decoder/conv2d_transpose_1/stack/2:output:01table_decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&table_decoder/conv2d_transpose_1/stack?
6table_decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6table_decoder/conv2d_transpose_1/strided_slice_1/stack?
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_1?
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_2?
0table_decoder/conv2d_transpose_1/strided_slice_1StridedSlice/table_decoder/conv2d_transpose_1/stack:output:0?table_decoder/conv2d_transpose_1/strided_slice_1/stack:output:0Atable_decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0Atable_decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0table_decoder/conv2d_transpose_1/strided_slice_1?
@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpItable_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02B
@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
1table_decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput/table_decoder/conv2d_transpose_1/stack:output:0Htable_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Ktable_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
23
1table_decoder/conv2d_transpose_1/conv2d_transpose?
7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp@table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?
(table_decoder/conv2d_transpose_1/BiasAddBiasAdd:table_decoder/conv2d_transpose_1/conv2d_transpose:output:0?table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2*
(table_decoder/conv2d_transpose_1/BiasAdd?
(table_decoder/conv2d_transpose_1/SoftmaxSoftmax1table_decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2*
(table_decoder/conv2d_transpose_1/Softmax?
(col_decoder/conv2d/Conv2D/ReadVariableOpReadVariableOp1col_decoder_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(col_decoder/conv2d/Conv2D/ReadVariableOp?
col_decoder/conv2d/Conv2DConv2Ddropout_1/Identity:output:00col_decoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
col_decoder/conv2d/Conv2D?
)col_decoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp2col_decoder_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)col_decoder/conv2d/BiasAdd/ReadVariableOp?
col_decoder/conv2d/BiasAddBiasAdd"col_decoder/conv2d/Conv2D:output:01col_decoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
col_decoder/conv2d/BiasAdd?
col_decoder/conv2d/ReluRelu#col_decoder/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
col_decoder/conv2d/Relu?
col_decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2!
col_decoder/up_sampling2d/Const?
!col_decoder/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2#
!col_decoder/up_sampling2d/Const_1?
col_decoder/up_sampling2d/mulMul(col_decoder/up_sampling2d/Const:output:0*col_decoder/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
col_decoder/up_sampling2d/mul?
/col_decoder/up_sampling2d/resize/ResizeBilinearResizeBilinear%col_decoder/conv2d/Relu:activations:0!col_decoder/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(21
/col_decoder/up_sampling2d/resize/ResizeBilinear?
%col_decoder/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%col_decoder/concatenate_2/concat/axis?
 col_decoder/concatenate_2/concatConcatV2block4_pool/MaxPool:output:0@col_decoder/up_sampling2d/resize/ResizeBilinear:resized_images:0.col_decoder/concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2"
 col_decoder/concatenate_2/concat?
!col_decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2#
!col_decoder/up_sampling2d_1/Const?
#col_decoder/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#col_decoder/up_sampling2d_1/Const_1?
col_decoder/up_sampling2d_1/mulMul*col_decoder/up_sampling2d_1/Const:output:0,col_decoder/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_1/mul?
1col_decoder/up_sampling2d_1/resize/ResizeBilinearResizeBilinear)col_decoder/concatenate_2/concat:output:0#col_decoder/up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(23
1col_decoder/up_sampling2d_1/resize/ResizeBilinear?
%col_decoder/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%col_decoder/concatenate_3/concat/axis?
 col_decoder/concatenate_3/concatConcatV2block3_pool/MaxPool:output:0Bcol_decoder/up_sampling2d_1/resize/ResizeBilinear:resized_images:0.col_decoder/concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2"
 col_decoder/concatenate_3/concat?
!col_decoder/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2#
!col_decoder/up_sampling2d_2/Const?
#col_decoder/up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#col_decoder/up_sampling2d_2/Const_1?
col_decoder/up_sampling2d_2/mulMul*col_decoder/up_sampling2d_2/Const:output:0,col_decoder/up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_2/mul?
8col_decoder/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor)col_decoder/concatenate_3/concat:output:0#col_decoder/up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2:
8col_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor?
!col_decoder/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2#
!col_decoder/up_sampling2d_3/Const?
#col_decoder/up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#col_decoder/up_sampling2d_3/Const_1?
col_decoder/up_sampling2d_3/mulMul*col_decoder/up_sampling2d_3/Const:output:0,col_decoder/up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_3/mul?
8col_decoder/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborIcol_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0#col_decoder/up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2:
8col_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor?
"col_decoder/conv2d_transpose/ShapeShapeIcol_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2$
"col_decoder/conv2d_transpose/Shape?
0col_decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0col_decoder/conv2d_transpose/strided_slice/stack?
2col_decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice/stack_1?
2col_decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice/stack_2?
*col_decoder/conv2d_transpose/strided_sliceStridedSlice+col_decoder/conv2d_transpose/Shape:output:09col_decoder/conv2d_transpose/strided_slice/stack:output:0;col_decoder/conv2d_transpose/strided_slice/stack_1:output:0;col_decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*col_decoder/conv2d_transpose/strided_slice?
$col_decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2&
$col_decoder/conv2d_transpose/stack/1?
$col_decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2&
$col_decoder/conv2d_transpose/stack/2?
$col_decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$col_decoder/conv2d_transpose/stack/3?
"col_decoder/conv2d_transpose/stackPack3col_decoder/conv2d_transpose/strided_slice:output:0-col_decoder/conv2d_transpose/stack/1:output:0-col_decoder/conv2d_transpose/stack/2:output:0-col_decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"col_decoder/conv2d_transpose/stack?
2col_decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2col_decoder/conv2d_transpose/strided_slice_1/stack?
4col_decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_1/stack_1?
4col_decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_1/stack_2?
,col_decoder/conv2d_transpose/strided_slice_1StridedSlice+col_decoder/conv2d_transpose/stack:output:0;col_decoder/conv2d_transpose/strided_slice_1/stack:output:0=col_decoder/conv2d_transpose/strided_slice_1/stack_1:output:0=col_decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,col_decoder/conv2d_transpose/strided_slice_1?
<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpEcol_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02>
<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?
-col_decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput+col_decoder/conv2d_transpose/stack:output:0Dcol_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0Icol_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2/
-col_decoder/conv2d_transpose/conv2d_transpose?
3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp<col_decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp?
$col_decoder/conv2d_transpose/BiasAddBiasAdd6col_decoder/conv2d_transpose/conv2d_transpose:output:0;col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2&
$col_decoder/conv2d_transpose/BiasAdd?
$col_decoder/conv2d_transpose/SoftmaxSoftmax-col_decoder/conv2d_transpose/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2&
$col_decoder/conv2d_transpose/Softmax?
IdentityIdentity.col_decoder/conv2d_transpose/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity2table_decoder/conv2d_transpose_1/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp&^block_6_conv_1/BiasAdd/ReadVariableOp%^block_6_conv_1/Conv2D/ReadVariableOp&^block_6_conv_2/BiasAdd/ReadVariableOp%^block_6_conv_2/Conv2D/ReadVariableOp*^col_decoder/conv2d/BiasAdd/ReadVariableOp)^col_decoder/conv2d/Conv2D/ReadVariableOp4^col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp=^col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp.^table_decoder/conv2d_1/BiasAdd/ReadVariableOp-^table_decoder/conv2d_1/Conv2D/ReadVariableOp.^table_decoder/conv2d_2/BiasAdd/ReadVariableOp-^table_decoder/conv2d_2/Conv2D/ReadVariableOp8^table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpA^table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp2N
%block_6_conv_1/BiasAdd/ReadVariableOp%block_6_conv_1/BiasAdd/ReadVariableOp2L
$block_6_conv_1/Conv2D/ReadVariableOp$block_6_conv_1/Conv2D/ReadVariableOp2N
%block_6_conv_2/BiasAdd/ReadVariableOp%block_6_conv_2/BiasAdd/ReadVariableOp2L
$block_6_conv_2/Conv2D/ReadVariableOp$block_6_conv_2/Conv2D/ReadVariableOp2V
)col_decoder/conv2d/BiasAdd/ReadVariableOp)col_decoder/conv2d/BiasAdd/ReadVariableOp2T
(col_decoder/conv2d/Conv2D/ReadVariableOp(col_decoder/conv2d/Conv2D/ReadVariableOp2j
3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp2|
<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^
-table_decoder/conv2d_1/BiasAdd/ReadVariableOp-table_decoder/conv2d_1/BiasAdd/ReadVariableOp2\
,table_decoder/conv2d_1/Conv2D/ReadVariableOp,table_decoder/conv2d_1/Conv2D/ReadVariableOp2^
-table_decoder/conv2d_2/BiasAdd/ReadVariableOp-table_decoder/conv2d_2/BiasAdd/ReadVariableOp2\
,table_decoder/conv2d_2/Conv2D/ReadVariableOp,table_decoder/conv2d_2/Conv2D/ReadVariableOp2r
7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv3_layer_call_and_return_conditional_losses_2032

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_1187

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_1797

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2	
MaxPooln
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_4239

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2	
MaxPooln
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
F
*__inference_block5_pool_layer_call_fn_4609

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_20592
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????22?:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
?
F__inference_block3_conv3_layer_call_and_return_conditional_losses_1884

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
+__inference_block3_conv1_layer_call_fn_4329

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_18502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv4_layer_call_and_return_conditional_losses_4480

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_4499

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????22?*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????22?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????dd?:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_2107

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_4620

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_block1_pool_layer_call_fn_4244

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_11872
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_1911

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????dd?*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????dd?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
+__inference_block5_conv1_layer_call_fn_4529

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_19982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
?
+__inference_block5_conv3_layer_call_fn_4569

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_20322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
F
*__inference_block3_pool_layer_call_fn_4404

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_12312
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_block3_conv2_layer_call_fn_4349

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_18672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?`
?
G__inference_table_decoder_layer_call_and_return_conditional_losses_4900
input_0
input_1
input_2C
'conv2d_1_conv2d_readvariableop_resource:??7
(conv2d_1_biasadd_readvariableop_resource:	?C
'conv2d_2_conv2d_readvariableop_resource:??7
(conv2d_2_biasadd_readvariableop_resource:	?V
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:?@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinput_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_1/BiasAdd|
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_1/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulconv2d_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout_2/dropout/Mul}
dropout_2/dropout/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_2/dropout/Mul_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_2/Relu
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_1?
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
%up_sampling2d_4/resize/ResizeBilinearResizeBilinearconv2d_1/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(2'
%up_sampling2d_4/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2input_16up_sampling2d_4/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2
concatenate/concat
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2
up_sampling2d_5/Const?
up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_5/Const_1?
up_sampling2d_5/mulMulup_sampling2d_5/Const:output:0 up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/mul?
%up_sampling2d_5/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(2'
%up_sampling2d_5/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2input_26up_sampling2d_5/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2
concatenate_1/concat
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2
up_sampling2d_6/Const?
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const_1?
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul?
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighbor
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
up_sampling2d_7/Const?
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const_1?
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul?
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighbor?
conv2d_transpose_1/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice{
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/1{
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/SoftmaxSoftmax#conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_1/Softmax?
IdentityIdentity$conv2d_transpose_1/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????:?????????22?:?????????dd?: : : : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input/0:YU
0
_output_shapes
:?????????22?
!
_user_specified_name	input/1:YU
0
_output_shapes
:?????????dd?
!
_user_specified_name	input/2
??
?(
?__inference_model_layer_call_and_return_conditional_losses_3991

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@?;
,block2_conv1_biasadd_readvariableop_resource:	?G
+block2_conv2_conv2d_readvariableop_resource:??;
,block2_conv2_biasadd_readvariableop_resource:	?G
+block3_conv1_conv2d_readvariableop_resource:??;
,block3_conv1_biasadd_readvariableop_resource:	?G
+block3_conv2_conv2d_readvariableop_resource:??;
,block3_conv2_biasadd_readvariableop_resource:	?G
+block3_conv3_conv2d_readvariableop_resource:??;
,block3_conv3_biasadd_readvariableop_resource:	?G
+block3_conv4_conv2d_readvariableop_resource:??;
,block3_conv4_biasadd_readvariableop_resource:	?G
+block4_conv1_conv2d_readvariableop_resource:??;
,block4_conv1_biasadd_readvariableop_resource:	?G
+block4_conv2_conv2d_readvariableop_resource:??;
,block4_conv2_biasadd_readvariableop_resource:	?G
+block4_conv3_conv2d_readvariableop_resource:??;
,block4_conv3_biasadd_readvariableop_resource:	?G
+block4_conv4_conv2d_readvariableop_resource:??;
,block4_conv4_biasadd_readvariableop_resource:	?G
+block5_conv1_conv2d_readvariableop_resource:??;
,block5_conv1_biasadd_readvariableop_resource:	?G
+block5_conv2_conv2d_readvariableop_resource:??;
,block5_conv2_biasadd_readvariableop_resource:	?G
+block5_conv3_conv2d_readvariableop_resource:??;
,block5_conv3_biasadd_readvariableop_resource:	?G
+block5_conv4_conv2d_readvariableop_resource:??;
,block5_conv4_biasadd_readvariableop_resource:	?I
-block_6_conv_1_conv2d_readvariableop_resource:??=
.block_6_conv_1_biasadd_readvariableop_resource:	?I
-block_6_conv_2_conv2d_readvariableop_resource:??=
.block_6_conv_2_biasadd_readvariableop_resource:	?Q
5table_decoder_conv2d_1_conv2d_readvariableop_resource:??E
6table_decoder_conv2d_1_biasadd_readvariableop_resource:	?Q
5table_decoder_conv2d_2_conv2d_readvariableop_resource:??E
6table_decoder_conv2d_2_biasadd_readvariableop_resource:	?d
Itable_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:?N
@table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource:M
1col_decoder_conv2d_conv2d_readvariableop_resource:??A
2col_decoder_conv2d_biasadd_readvariableop_resource:	?`
Ecol_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource:?J
<col_decoder_conv2d_transpose_biasadd_readvariableop_resource:
identity

identity_1??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block3_conv4/BiasAdd/ReadVariableOp?"block3_conv4/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block4_conv4/BiasAdd/ReadVariableOp?"block4_conv4/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?#block5_conv4/BiasAdd/ReadVariableOp?"block5_conv4/Conv2D/ReadVariableOp?%block_6_conv_1/BiasAdd/ReadVariableOp?$block_6_conv_1/Conv2D/ReadVariableOp?%block_6_conv_2/BiasAdd/ReadVariableOp?$block_6_conv_2/Conv2D/ReadVariableOp?)col_decoder/conv2d/BiasAdd/ReadVariableOp?(col_decoder/conv2d/Conv2D/ReadVariableOp?3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp?<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?-table_decoder/conv2d_1/BiasAdd/ReadVariableOp?,table_decoder/conv2d_1/Conv2D/ReadVariableOp?-table_decoder/conv2d_2/BiasAdd/ReadVariableOp?,table_decoder/conv2d_2/Conv2D/ReadVariableOp?7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv1/Relu?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*2
_output_shapes 
:????????????*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block3_conv1/Relu?
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block3_conv2/Conv2D?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block3_conv2/BiasAdd?
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block3_conv2/Relu?
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block3_conv3/Conv2D?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block3_conv3/BiasAdd?
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block3_conv3/Relu?
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block3_conv4/Conv2D/ReadVariableOp?
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
block3_conv4/Conv2D?
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block3_conv4/BiasAdd/ReadVariableOp?
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
block3_conv4/BiasAdd?
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
block3_conv4/Relu?
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:?????????dd?*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
block4_conv1/Conv2D?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv1/BiasAdd?
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv1/Relu?
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
block4_conv2/Conv2D?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv2/BiasAdd?
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv2/Relu?
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
block4_conv3/Conv2D?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv3/BiasAdd?
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv3/Relu?
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block4_conv4/Conv2D/ReadVariableOp?
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
block4_conv4/Conv2D?
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block4_conv4/BiasAdd/ReadVariableOp?
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv4/BiasAdd?
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
block4_conv4/Relu?
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:?????????22?*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
block5_conv1/Conv2D?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
block5_conv1/BiasAdd?
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
block5_conv1/Relu?
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv2/Conv2D/ReadVariableOp?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
block5_conv2/Conv2D?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
block5_conv2/BiasAdd?
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
block5_conv2/Relu?
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv3/Conv2D/ReadVariableOp?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
block5_conv3/Conv2D?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
block5_conv3/BiasAdd?
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
block5_conv3/Relu?
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02$
"block5_conv4/Conv2D/ReadVariableOp?
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
block5_conv4/Conv2D?
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#block5_conv4/BiasAdd/ReadVariableOp?
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
block5_conv4/BiasAdd?
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
block5_conv4/Relu?
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool?
$block_6_conv_1/Conv2D/ReadVariableOpReadVariableOp-block_6_conv_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$block_6_conv_1/Conv2D/ReadVariableOp?
block_6_conv_1/Conv2DConv2Dblock5_pool/MaxPool:output:0,block_6_conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
block_6_conv_1/Conv2D?
%block_6_conv_1/BiasAdd/ReadVariableOpReadVariableOp.block_6_conv_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%block_6_conv_1/BiasAdd/ReadVariableOp?
block_6_conv_1/BiasAddBiasAddblock_6_conv_1/Conv2D:output:0-block_6_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block_6_conv_1/BiasAdd?
block_6_conv_1/ReluRelublock_6_conv_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block_6_conv_1/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMul!block_6_conv_1/Relu:activations:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/dropout/Mul
dropout/dropout/ShapeShape!block_6_conv_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/dropout/Mul_1?
$block_6_conv_2/Conv2D/ReadVariableOpReadVariableOp-block_6_conv_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$block_6_conv_2/Conv2D/ReadVariableOp?
block_6_conv_2/Conv2DConv2Ddropout/dropout/Mul_1:z:0,block_6_conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
block_6_conv_2/Conv2D?
%block_6_conv_2/BiasAdd/ReadVariableOpReadVariableOp.block_6_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%block_6_conv_2/BiasAdd/ReadVariableOp?
block_6_conv_2/BiasAddBiasAddblock_6_conv_2/Conv2D:output:0-block_6_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block_6_conv_2/BiasAdd?
block_6_conv_2/ReluRelublock_6_conv_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block_6_conv_2/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul!block_6_conv_2/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape!block_6_conv_2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_1/dropout/Mul_1?
,table_decoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5table_decoder_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02.
,table_decoder/conv2d_1/Conv2D/ReadVariableOp?
table_decoder/conv2d_1/Conv2DConv2Ddropout_1/dropout/Mul_1:z:04table_decoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
table_decoder/conv2d_1/Conv2D?
-table_decoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6table_decoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-table_decoder/conv2d_1/BiasAdd/ReadVariableOp?
table_decoder/conv2d_1/BiasAddBiasAdd&table_decoder/conv2d_1/Conv2D:output:05table_decoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
table_decoder/conv2d_1/BiasAdd?
table_decoder/conv2d_1/ReluRelu'table_decoder/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
table_decoder/conv2d_1/Relu?
%table_decoder/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%table_decoder/dropout_2/dropout/Const?
#table_decoder/dropout_2/dropout/MulMul)table_decoder/conv2d_1/Relu:activations:0.table_decoder/dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2%
#table_decoder/dropout_2/dropout/Mul?
%table_decoder/dropout_2/dropout/ShapeShape)table_decoder/conv2d_1/Relu:activations:0*
T0*
_output_shapes
:2'
%table_decoder/dropout_2/dropout/Shape?
<table_decoder/dropout_2/dropout/random_uniform/RandomUniformRandomUniform.table_decoder/dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02>
<table_decoder/dropout_2/dropout/random_uniform/RandomUniform?
.table_decoder/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>20
.table_decoder/dropout_2/dropout/GreaterEqual/y?
,table_decoder/dropout_2/dropout/GreaterEqualGreaterEqualEtable_decoder/dropout_2/dropout/random_uniform/RandomUniform:output:07table_decoder/dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2.
,table_decoder/dropout_2/dropout/GreaterEqual?
$table_decoder/dropout_2/dropout/CastCast0table_decoder/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2&
$table_decoder/dropout_2/dropout/Cast?
%table_decoder/dropout_2/dropout/Mul_1Mul'table_decoder/dropout_2/dropout/Mul:z:0(table_decoder/dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2'
%table_decoder/dropout_2/dropout/Mul_1?
,table_decoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5table_decoder_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02.
,table_decoder/conv2d_2/Conv2D/ReadVariableOp?
table_decoder/conv2d_2/Conv2DConv2D)table_decoder/dropout_2/dropout/Mul_1:z:04table_decoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
table_decoder/conv2d_2/Conv2D?
-table_decoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6table_decoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-table_decoder/conv2d_2/BiasAdd/ReadVariableOp?
table_decoder/conv2d_2/BiasAddBiasAdd&table_decoder/conv2d_2/Conv2D:output:05table_decoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
table_decoder/conv2d_2/BiasAdd?
table_decoder/conv2d_2/ReluRelu'table_decoder/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
table_decoder/conv2d_2/Relu?
#table_decoder/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#table_decoder/up_sampling2d_4/Const?
%table_decoder/up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2'
%table_decoder/up_sampling2d_4/Const_1?
!table_decoder/up_sampling2d_4/mulMul,table_decoder/up_sampling2d_4/Const:output:0.table_decoder/up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_4/mul?
3table_decoder/up_sampling2d_4/resize/ResizeBilinearResizeBilinear)table_decoder/conv2d_1/Relu:activations:0%table_decoder/up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(25
3table_decoder/up_sampling2d_4/resize/ResizeBilinear?
%table_decoder/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%table_decoder/concatenate/concat/axis?
 table_decoder/concatenate/concatConcatV2block4_pool/MaxPool:output:0Dtable_decoder/up_sampling2d_4/resize/ResizeBilinear:resized_images:0.table_decoder/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2"
 table_decoder/concatenate/concat?
#table_decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2%
#table_decoder/up_sampling2d_5/Const?
%table_decoder/up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2'
%table_decoder/up_sampling2d_5/Const_1?
!table_decoder/up_sampling2d_5/mulMul,table_decoder/up_sampling2d_5/Const:output:0.table_decoder/up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_5/mul?
3table_decoder/up_sampling2d_5/resize/ResizeBilinearResizeBilinear)table_decoder/concatenate/concat:output:0%table_decoder/up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(25
3table_decoder/up_sampling2d_5/resize/ResizeBilinear?
'table_decoder/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'table_decoder/concatenate_1/concat/axis?
"table_decoder/concatenate_1/concatConcatV2block3_pool/MaxPool:output:0Dtable_decoder/up_sampling2d_5/resize/ResizeBilinear:resized_images:00table_decoder/concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2$
"table_decoder/concatenate_1/concat?
#table_decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2%
#table_decoder/up_sampling2d_6/Const?
%table_decoder/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2'
%table_decoder/up_sampling2d_6/Const_1?
!table_decoder/up_sampling2d_6/mulMul,table_decoder/up_sampling2d_6/Const:output:0.table_decoder/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_6/mul?
:table_decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor+table_decoder/concatenate_1/concat:output:0%table_decoder/up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2<
:table_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor?
#table_decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2%
#table_decoder/up_sampling2d_7/Const?
%table_decoder/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2'
%table_decoder/up_sampling2d_7/Const_1?
!table_decoder/up_sampling2d_7/mulMul,table_decoder/up_sampling2d_7/Const:output:0.table_decoder/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2#
!table_decoder/up_sampling2d_7/mul?
:table_decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborKtable_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0%table_decoder/up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2<
:table_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor?
&table_decoder/conv2d_transpose_1/ShapeShapeKtable_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2(
&table_decoder/conv2d_transpose_1/Shape?
4table_decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4table_decoder/conv2d_transpose_1/strided_slice/stack?
6table_decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice/stack_1?
6table_decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6table_decoder/conv2d_transpose_1/strided_slice/stack_2?
.table_decoder/conv2d_transpose_1/strided_sliceStridedSlice/table_decoder/conv2d_transpose_1/Shape:output:0=table_decoder/conv2d_transpose_1/strided_slice/stack:output:0?table_decoder/conv2d_transpose_1/strided_slice/stack_1:output:0?table_decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.table_decoder/conv2d_transpose_1/strided_slice?
(table_decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2*
(table_decoder/conv2d_transpose_1/stack/1?
(table_decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2*
(table_decoder/conv2d_transpose_1/stack/2?
(table_decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(table_decoder/conv2d_transpose_1/stack/3?
&table_decoder/conv2d_transpose_1/stackPack7table_decoder/conv2d_transpose_1/strided_slice:output:01table_decoder/conv2d_transpose_1/stack/1:output:01table_decoder/conv2d_transpose_1/stack/2:output:01table_decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&table_decoder/conv2d_transpose_1/stack?
6table_decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6table_decoder/conv2d_transpose_1/strided_slice_1/stack?
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_1?
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8table_decoder/conv2d_transpose_1/strided_slice_1/stack_2?
0table_decoder/conv2d_transpose_1/strided_slice_1StridedSlice/table_decoder/conv2d_transpose_1/stack:output:0?table_decoder/conv2d_transpose_1/strided_slice_1/stack:output:0Atable_decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0Atable_decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0table_decoder/conv2d_transpose_1/strided_slice_1?
@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpItable_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02B
@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
1table_decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput/table_decoder/conv2d_transpose_1/stack:output:0Htable_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Ktable_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
23
1table_decoder/conv2d_transpose_1/conv2d_transpose?
7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp@table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?
(table_decoder/conv2d_transpose_1/BiasAddBiasAdd:table_decoder/conv2d_transpose_1/conv2d_transpose:output:0?table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2*
(table_decoder/conv2d_transpose_1/BiasAdd?
(table_decoder/conv2d_transpose_1/SoftmaxSoftmax1table_decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2*
(table_decoder/conv2d_transpose_1/Softmax?
(col_decoder/conv2d/Conv2D/ReadVariableOpReadVariableOp1col_decoder_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(col_decoder/conv2d/Conv2D/ReadVariableOp?
col_decoder/conv2d/Conv2DConv2Ddropout_1/dropout/Mul_1:z:00col_decoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
col_decoder/conv2d/Conv2D?
)col_decoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp2col_decoder_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)col_decoder/conv2d/BiasAdd/ReadVariableOp?
col_decoder/conv2d/BiasAddBiasAdd"col_decoder/conv2d/Conv2D:output:01col_decoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
col_decoder/conv2d/BiasAdd?
col_decoder/conv2d/ReluRelu#col_decoder/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
col_decoder/conv2d/Relu?
col_decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2!
col_decoder/up_sampling2d/Const?
!col_decoder/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2#
!col_decoder/up_sampling2d/Const_1?
col_decoder/up_sampling2d/mulMul(col_decoder/up_sampling2d/Const:output:0*col_decoder/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
col_decoder/up_sampling2d/mul?
/col_decoder/up_sampling2d/resize/ResizeBilinearResizeBilinear%col_decoder/conv2d/Relu:activations:0!col_decoder/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(21
/col_decoder/up_sampling2d/resize/ResizeBilinear?
%col_decoder/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%col_decoder/concatenate_2/concat/axis?
 col_decoder/concatenate_2/concatConcatV2block4_pool/MaxPool:output:0@col_decoder/up_sampling2d/resize/ResizeBilinear:resized_images:0.col_decoder/concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2"
 col_decoder/concatenate_2/concat?
!col_decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2#
!col_decoder/up_sampling2d_1/Const?
#col_decoder/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#col_decoder/up_sampling2d_1/Const_1?
col_decoder/up_sampling2d_1/mulMul*col_decoder/up_sampling2d_1/Const:output:0,col_decoder/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_1/mul?
1col_decoder/up_sampling2d_1/resize/ResizeBilinearResizeBilinear)col_decoder/concatenate_2/concat:output:0#col_decoder/up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(23
1col_decoder/up_sampling2d_1/resize/ResizeBilinear?
%col_decoder/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%col_decoder/concatenate_3/concat/axis?
 col_decoder/concatenate_3/concatConcatV2block3_pool/MaxPool:output:0Bcol_decoder/up_sampling2d_1/resize/ResizeBilinear:resized_images:0.col_decoder/concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2"
 col_decoder/concatenate_3/concat?
!col_decoder/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2#
!col_decoder/up_sampling2d_2/Const?
#col_decoder/up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#col_decoder/up_sampling2d_2/Const_1?
col_decoder/up_sampling2d_2/mulMul*col_decoder/up_sampling2d_2/Const:output:0,col_decoder/up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_2/mul?
8col_decoder/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor)col_decoder/concatenate_3/concat:output:0#col_decoder/up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2:
8col_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor?
!col_decoder/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2#
!col_decoder/up_sampling2d_3/Const?
#col_decoder/up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#col_decoder/up_sampling2d_3/Const_1?
col_decoder/up_sampling2d_3/mulMul*col_decoder/up_sampling2d_3/Const:output:0,col_decoder/up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:2!
col_decoder/up_sampling2d_3/mul?
8col_decoder/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborIcol_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0#col_decoder/up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2:
8col_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor?
"col_decoder/conv2d_transpose/ShapeShapeIcol_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2$
"col_decoder/conv2d_transpose/Shape?
0col_decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0col_decoder/conv2d_transpose/strided_slice/stack?
2col_decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice/stack_1?
2col_decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2col_decoder/conv2d_transpose/strided_slice/stack_2?
*col_decoder/conv2d_transpose/strided_sliceStridedSlice+col_decoder/conv2d_transpose/Shape:output:09col_decoder/conv2d_transpose/strided_slice/stack:output:0;col_decoder/conv2d_transpose/strided_slice/stack_1:output:0;col_decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*col_decoder/conv2d_transpose/strided_slice?
$col_decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2&
$col_decoder/conv2d_transpose/stack/1?
$col_decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2&
$col_decoder/conv2d_transpose/stack/2?
$col_decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$col_decoder/conv2d_transpose/stack/3?
"col_decoder/conv2d_transpose/stackPack3col_decoder/conv2d_transpose/strided_slice:output:0-col_decoder/conv2d_transpose/stack/1:output:0-col_decoder/conv2d_transpose/stack/2:output:0-col_decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"col_decoder/conv2d_transpose/stack?
2col_decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2col_decoder/conv2d_transpose/strided_slice_1/stack?
4col_decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_1/stack_1?
4col_decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4col_decoder/conv2d_transpose/strided_slice_1/stack_2?
,col_decoder/conv2d_transpose/strided_slice_1StridedSlice+col_decoder/conv2d_transpose/stack:output:0;col_decoder/conv2d_transpose/strided_slice_1/stack:output:0=col_decoder/conv2d_transpose/strided_slice_1/stack_1:output:0=col_decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,col_decoder/conv2d_transpose/strided_slice_1?
<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpEcol_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02>
<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?
-col_decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput+col_decoder/conv2d_transpose/stack:output:0Dcol_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0Icol_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2/
-col_decoder/conv2d_transpose/conv2d_transpose?
3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp<col_decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp?
$col_decoder/conv2d_transpose/BiasAddBiasAdd6col_decoder/conv2d_transpose/conv2d_transpose:output:0;col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2&
$col_decoder/conv2d_transpose/BiasAdd?
$col_decoder/conv2d_transpose/SoftmaxSoftmax-col_decoder/conv2d_transpose/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2&
$col_decoder/conv2d_transpose/Softmax?
IdentityIdentity.col_decoder/conv2d_transpose/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity2table_decoder/conv2d_transpose_1/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp&^block_6_conv_1/BiasAdd/ReadVariableOp%^block_6_conv_1/Conv2D/ReadVariableOp&^block_6_conv_2/BiasAdd/ReadVariableOp%^block_6_conv_2/Conv2D/ReadVariableOp*^col_decoder/conv2d/BiasAdd/ReadVariableOp)^col_decoder/conv2d/Conv2D/ReadVariableOp4^col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp=^col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp.^table_decoder/conv2d_1/BiasAdd/ReadVariableOp-^table_decoder/conv2d_1/Conv2D/ReadVariableOp.^table_decoder/conv2d_2/BiasAdd/ReadVariableOp-^table_decoder/conv2d_2/Conv2D/ReadVariableOp8^table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpA^table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp2N
%block_6_conv_1/BiasAdd/ReadVariableOp%block_6_conv_1/BiasAdd/ReadVariableOp2L
$block_6_conv_1/Conv2D/ReadVariableOp$block_6_conv_1/Conv2D/ReadVariableOp2N
%block_6_conv_2/BiasAdd/ReadVariableOp%block_6_conv_2/BiasAdd/ReadVariableOp2L
$block_6_conv_2/Conv2D/ReadVariableOp$block_6_conv_2/Conv2D/ReadVariableOp2V
)col_decoder/conv2d/BiasAdd/ReadVariableOp)col_decoder/conv2d/BiasAdd/ReadVariableOp2T
(col_decoder/conv2d/Conv2D/ReadVariableOp(col_decoder/conv2d/Conv2D/ReadVariableOp2j
3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp3col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp2|
<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp<col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^
-table_decoder/conv2d_1/BiasAdd/ReadVariableOp-table_decoder/conv2d_1/BiasAdd/ReadVariableOp2\
,table_decoder/conv2d_1/Conv2D/ReadVariableOp,table_decoder/conv2d_1/Conv2D/ReadVariableOp2^
-table_decoder/conv2d_2/BiasAdd/ReadVariableOp-table_decoder/conv2d_2/BiasAdd/ReadVariableOp2\
,table_decoder/conv2d_2/Conv2D/ReadVariableOp,table_decoder/conv2d_2/Conv2D/ReadVariableOp2r
7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp7table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp@table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?_
?
G__inference_table_decoder_layer_call_and_return_conditional_losses_2454	
input
input_1
input_2C
'conv2d_1_conv2d_readvariableop_resource:??7
(conv2d_1_biasadd_readvariableop_resource:	?C
'conv2d_2_conv2d_readvariableop_resource:??7
(conv2d_2_biasadd_readvariableop_resource:	?V
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:?@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinput&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_1/BiasAdd|
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_1/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulconv2d_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout_2/dropout/Mul}
dropout_2/dropout/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_2/dropout/Mul_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_2/Relu
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_1?
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
%up_sampling2d_4/resize/ResizeBilinearResizeBilinearconv2d_1/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(2'
%up_sampling2d_4/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2input_16up_sampling2d_4/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2
concatenate/concat
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2
up_sampling2d_5/Const?
up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_5/Const_1?
up_sampling2d_5/mulMulup_sampling2d_5/Const:output:0 up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/mul?
%up_sampling2d_5/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(2'
%up_sampling2d_5/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2input_26up_sampling2d_5/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2
concatenate_1/concat
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2
up_sampling2d_6/Const?
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const_1?
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul?
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighbor
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
up_sampling2d_7/Const?
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const_1?
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul?
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighbor?
conv2d_transpose_1/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice{
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/1{
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/SoftmaxSoftmax#conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_1/Softmax?
IdentityIdentity$conv2d_transpose_1/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????:?????????22?:?????????dd?: : : : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:W S
0
_output_shapes
:??????????

_user_specified_nameinput:WS
0
_output_shapes
:?????????22?

_user_specified_nameinput:WS
0
_output_shapes
:?????????dd?

_user_specified_nameinput
?
_
&__inference_dropout_layer_call_fn_4656

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_25222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_4220

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
F__inference_block5_conv3_layer_call_and_return_conditional_losses_4560

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_5001

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
B
&__inference_dropout_layer_call_fn_4651

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_20832
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block3_conv2_layer_call_and_return_conditional_losses_1867

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
J
.__inference_up_sampling2d_1_layer_call_fn_4972

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_13402
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_1787

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
-__inference_block_6_conv_1_layer_call_fn_4629

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_20722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block3_conv2_layer_call_and_return_conditional_losses_4340

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv1_layer_call_and_return_conditional_losses_4520

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1340

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_4967

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_1304

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?V
?
G__inference_table_decoder_layer_call_and_return_conditional_losses_2170	
input
input_1
input_2C
'conv2d_1_conv2d_readvariableop_resource:??7
(conv2d_1_biasadd_readvariableop_resource:	?C
'conv2d_2_conv2d_readvariableop_resource:??7
(conv2d_2_biasadd_readvariableop_resource:	?V
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:?@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinput&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_1/BiasAdd|
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_1/Relu?
dropout_2/IdentityIdentityconv2d_1/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_2/Identity?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Ddropout_2/Identity:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_2/Relu
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_1?
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
%up_sampling2d_4/resize/ResizeBilinearResizeBilinearconv2d_1/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(2'
%up_sampling2d_4/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2input_16up_sampling2d_4/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2
concatenate/concat
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2
up_sampling2d_5/Const?
up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_5/Const_1?
up_sampling2d_5/mulMulup_sampling2d_5/Const:output:0 up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/mul?
%up_sampling2d_5/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(2'
%up_sampling2d_5/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2input_26up_sampling2d_5/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2
concatenate_1/concat
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2
up_sampling2d_6/Const?
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const_1?
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul?
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighbor
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
up_sampling2d_7/Const?
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const_1?
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul?
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighbor?
conv2d_transpose_1/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice{
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/1{
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/SoftmaxSoftmax#conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_1/Softmax?
IdentityIdentity$conv2d_transpose_1/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????:?????????22?:?????????dd?: : : : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:W S
0
_output_shapes
:??????????

_user_specified_nameinput:WS
0
_output_shapes
:?????????22?

_user_specified_nameinput:WS
0
_output_shapes
:?????????dd?

_user_specified_nameinput
?
?
+__inference_block4_conv1_layer_call_fn_4429

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_19242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
?
F__inference_block4_conv1_layer_call_and_return_conditional_losses_1924

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
?
F__inference_block2_conv2_layer_call_and_return_conditional_losses_4280

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
+__inference_block5_conv2_layer_call_fn_4549

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_20152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
?
+__inference_block5_conv4_layer_call_fn_4589

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_20492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_1572

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_2083

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv2_layer_call_and_return_conditional_losses_4440

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?V
?
G__inference_table_decoder_layer_call_and_return_conditional_losses_4832
input_0
input_1
input_2C
'conv2d_1_conv2d_readvariableop_resource:??7
(conv2d_1_biasadd_readvariableop_resource:	?C
'conv2d_2_conv2d_readvariableop_resource:??7
(conv2d_2_biasadd_readvariableop_resource:	?V
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:?@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinput_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_1/BiasAdd|
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_1/Relu?
dropout_2/IdentityIdentityconv2d_1/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_2/Identity?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Ddropout_2/Identity:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_2/Relu
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_1?
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
%up_sampling2d_4/resize/ResizeBilinearResizeBilinearconv2d_1/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(2'
%up_sampling2d_4/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2input_16up_sampling2d_4/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2
concatenate/concat
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2
up_sampling2d_5/Const?
up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_5/Const_1?
up_sampling2d_5/mulMulup_sampling2d_5/Const:output:0 up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/mul?
%up_sampling2d_5/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(2'
%up_sampling2d_5/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2input_26up_sampling2d_5/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2
concatenate_1/concat
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2
up_sampling2d_6/Const?
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const_1?
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul?
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighbor
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
up_sampling2d_7/Const?
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const_1?
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul?
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighbor?
conv2d_transpose_1/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice{
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/1{
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/SoftmaxSoftmax#conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_1/Softmax?
IdentityIdentity$conv2d_transpose_1/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????:?????????22?:?????????dd?: : : : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input/0:YU
0
_output_shapes
:?????????22?
!
_user_specified_name	input/1:YU
0
_output_shapes
:?????????dd?
!
_user_specified_name	input/2
?
e
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1376

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_up_sampling2d_2_layer_call_fn_4989

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_13762
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv1_layer_call_and_return_conditional_losses_1998

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
a
(__inference_dropout_1_layer_call_fn_4703

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_24892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv4_layer_call_and_return_conditional_losses_1975

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_4599

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????22?:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
?
F__inference_block3_conv4_layer_call_and_return_conditional_losses_4380

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_3490
input_layer!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?&

unknown_33:??

unknown_34:	?&

unknown_35:??

unknown_36:	?&

unknown_37:??

unknown_38:	?%

unknown_39:?

unknown_40:&

unknown_41:??

unknown_42:	?%

unknown_43:?

unknown_44:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::???????????:???????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_11782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameInput_Layer
?
?
F__inference_block4_conv3_layer_call_and_return_conditional_losses_4460

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
?
F__inference_block2_conv2_layer_call_and_return_conditional_losses_1827

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_1_layer_call_fn_4698

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_21072
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_1209

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_1837

inputs
identity?
MaxPoolMaxPoolinputs*2
_output_shapes 
:????????????*
ksize
*
paddingVALID*
strides
2	
MaxPoolo
IdentityIdentityMaxPool:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1536

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_1412

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_5061

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_1608

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv2_layer_call_and_return_conditional_losses_2015

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_5112

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_table_decoder_layer_call_fn_4919
input_0
input_1
input_2#
unknown:??
	unknown_0:	?%
	unknown_1:??
	unknown_2:	?$
	unknown_3:?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_21702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????:?????????22?:?????????dd?: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input/0:YU
0
_output_shapes
:?????????22?
!
_user_specified_name	input/1:YU
0
_output_shapes
:?????????dd?
!
_user_specified_name	input/2
?
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_1231

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_block4_conv3_layer_call_fn_4469

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_19582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
F
*__inference_block4_pool_layer_call_fn_4509

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_19852
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????22?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????dd?:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_2489

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_4594

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_up_sampling2d_5_layer_call_fn_5083

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_15722
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv2_layer_call_and_return_conditional_losses_4540

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????22?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????22?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
?
F__inference_block3_conv1_layer_call_and_return_conditional_losses_4320

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
+__inference_block4_conv2_layer_call_fn_4449

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_19412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_4693

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_up_sampling2d_6_layer_call_fn_5100

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_16082
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?&
?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1470

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SoftmaxSoftmaxBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?,
__inference__wrapped_model_1178
input_layerK
1model_block1_conv1_conv2d_readvariableop_resource:@@
2model_block1_conv1_biasadd_readvariableop_resource:@K
1model_block1_conv2_conv2d_readvariableop_resource:@@@
2model_block1_conv2_biasadd_readvariableop_resource:@L
1model_block2_conv1_conv2d_readvariableop_resource:@?A
2model_block2_conv1_biasadd_readvariableop_resource:	?M
1model_block2_conv2_conv2d_readvariableop_resource:??A
2model_block2_conv2_biasadd_readvariableop_resource:	?M
1model_block3_conv1_conv2d_readvariableop_resource:??A
2model_block3_conv1_biasadd_readvariableop_resource:	?M
1model_block3_conv2_conv2d_readvariableop_resource:??A
2model_block3_conv2_biasadd_readvariableop_resource:	?M
1model_block3_conv3_conv2d_readvariableop_resource:??A
2model_block3_conv3_biasadd_readvariableop_resource:	?M
1model_block3_conv4_conv2d_readvariableop_resource:??A
2model_block3_conv4_biasadd_readvariableop_resource:	?M
1model_block4_conv1_conv2d_readvariableop_resource:??A
2model_block4_conv1_biasadd_readvariableop_resource:	?M
1model_block4_conv2_conv2d_readvariableop_resource:??A
2model_block4_conv2_biasadd_readvariableop_resource:	?M
1model_block4_conv3_conv2d_readvariableop_resource:??A
2model_block4_conv3_biasadd_readvariableop_resource:	?M
1model_block4_conv4_conv2d_readvariableop_resource:??A
2model_block4_conv4_biasadd_readvariableop_resource:	?M
1model_block5_conv1_conv2d_readvariableop_resource:??A
2model_block5_conv1_biasadd_readvariableop_resource:	?M
1model_block5_conv2_conv2d_readvariableop_resource:??A
2model_block5_conv2_biasadd_readvariableop_resource:	?M
1model_block5_conv3_conv2d_readvariableop_resource:??A
2model_block5_conv3_biasadd_readvariableop_resource:	?M
1model_block5_conv4_conv2d_readvariableop_resource:??A
2model_block5_conv4_biasadd_readvariableop_resource:	?O
3model_block_6_conv_1_conv2d_readvariableop_resource:??C
4model_block_6_conv_1_biasadd_readvariableop_resource:	?O
3model_block_6_conv_2_conv2d_readvariableop_resource:??C
4model_block_6_conv_2_biasadd_readvariableop_resource:	?W
;model_table_decoder_conv2d_1_conv2d_readvariableop_resource:??K
<model_table_decoder_conv2d_1_biasadd_readvariableop_resource:	?W
;model_table_decoder_conv2d_2_conv2d_readvariableop_resource:??K
<model_table_decoder_conv2d_2_biasadd_readvariableop_resource:	?j
Omodel_table_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:?T
Fmodel_table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource:S
7model_col_decoder_conv2d_conv2d_readvariableop_resource:??G
8model_col_decoder_conv2d_biasadd_readvariableop_resource:	?f
Kmodel_col_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource:?P
Bmodel_col_decoder_conv2d_transpose_biasadd_readvariableop_resource:
identity

identity_1??)model/block1_conv1/BiasAdd/ReadVariableOp?(model/block1_conv1/Conv2D/ReadVariableOp?)model/block1_conv2/BiasAdd/ReadVariableOp?(model/block1_conv2/Conv2D/ReadVariableOp?)model/block2_conv1/BiasAdd/ReadVariableOp?(model/block2_conv1/Conv2D/ReadVariableOp?)model/block2_conv2/BiasAdd/ReadVariableOp?(model/block2_conv2/Conv2D/ReadVariableOp?)model/block3_conv1/BiasAdd/ReadVariableOp?(model/block3_conv1/Conv2D/ReadVariableOp?)model/block3_conv2/BiasAdd/ReadVariableOp?(model/block3_conv2/Conv2D/ReadVariableOp?)model/block3_conv3/BiasAdd/ReadVariableOp?(model/block3_conv3/Conv2D/ReadVariableOp?)model/block3_conv4/BiasAdd/ReadVariableOp?(model/block3_conv4/Conv2D/ReadVariableOp?)model/block4_conv1/BiasAdd/ReadVariableOp?(model/block4_conv1/Conv2D/ReadVariableOp?)model/block4_conv2/BiasAdd/ReadVariableOp?(model/block4_conv2/Conv2D/ReadVariableOp?)model/block4_conv3/BiasAdd/ReadVariableOp?(model/block4_conv3/Conv2D/ReadVariableOp?)model/block4_conv4/BiasAdd/ReadVariableOp?(model/block4_conv4/Conv2D/ReadVariableOp?)model/block5_conv1/BiasAdd/ReadVariableOp?(model/block5_conv1/Conv2D/ReadVariableOp?)model/block5_conv2/BiasAdd/ReadVariableOp?(model/block5_conv2/Conv2D/ReadVariableOp?)model/block5_conv3/BiasAdd/ReadVariableOp?(model/block5_conv3/Conv2D/ReadVariableOp?)model/block5_conv4/BiasAdd/ReadVariableOp?(model/block5_conv4/Conv2D/ReadVariableOp?+model/block_6_conv_1/BiasAdd/ReadVariableOp?*model/block_6_conv_1/Conv2D/ReadVariableOp?+model/block_6_conv_2/BiasAdd/ReadVariableOp?*model/block_6_conv_2/Conv2D/ReadVariableOp?/model/col_decoder/conv2d/BiasAdd/ReadVariableOp?.model/col_decoder/conv2d/Conv2D/ReadVariableOp?9model/col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp?Bmodel/col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?3model/table_decoder/conv2d_1/BiasAdd/ReadVariableOp?2model/table_decoder/conv2d_1/Conv2D/ReadVariableOp?3model/table_decoder/conv2d_2/BiasAdd/ReadVariableOp?2model/table_decoder/conv2d_2/Conv2D/ReadVariableOp?=model/table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?Fmodel/table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
(model/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(model/block1_conv1/Conv2D/ReadVariableOp?
model/block1_conv1/Conv2DConv2Dinput_layer0model/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
model/block1_conv1/Conv2D?
)model/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model/block1_conv1/BiasAdd/ReadVariableOp?
model/block1_conv1/BiasAddBiasAdd"model/block1_conv1/Conv2D:output:01model/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
model/block1_conv1/BiasAdd?
model/block1_conv1/ReluRelu#model/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
model/block1_conv1/Relu?
(model/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(model/block1_conv2/Conv2D/ReadVariableOp?
model/block1_conv2/Conv2DConv2D%model/block1_conv1/Relu:activations:00model/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
model/block1_conv2/Conv2D?
)model/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model/block1_conv2/BiasAdd/ReadVariableOp?
model/block1_conv2/BiasAddBiasAdd"model/block1_conv2/Conv2D:output:01model/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
model/block1_conv2/BiasAdd?
model/block1_conv2/ReluRelu#model/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
model/block1_conv2/Relu?
model/block1_pool/MaxPoolMaxPool%model/block1_conv2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
model/block1_pool/MaxPool?
(model/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02*
(model/block2_conv1/Conv2D/ReadVariableOp?
model/block2_conv1/Conv2DConv2D"model/block1_pool/MaxPool:output:00model/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model/block2_conv1/Conv2D?
)model/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block2_conv1/BiasAdd/ReadVariableOp?
model/block2_conv1/BiasAddBiasAdd"model/block2_conv1/Conv2D:output:01model/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
model/block2_conv1/BiasAdd?
model/block2_conv1/ReluRelu#model/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
model/block2_conv1/Relu?
(model/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block2_conv2/Conv2D/ReadVariableOp?
model/block2_conv2/Conv2DConv2D%model/block2_conv1/Relu:activations:00model/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model/block2_conv2/Conv2D?
)model/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block2_conv2/BiasAdd/ReadVariableOp?
model/block2_conv2/BiasAddBiasAdd"model/block2_conv2/Conv2D:output:01model/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
model/block2_conv2/BiasAdd?
model/block2_conv2/ReluRelu#model/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
model/block2_conv2/Relu?
model/block2_pool/MaxPoolMaxPool%model/block2_conv2/Relu:activations:0*2
_output_shapes 
:????????????*
ksize
*
paddingVALID*
strides
2
model/block2_pool/MaxPool?
(model/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block3_conv1/Conv2D/ReadVariableOp?
model/block3_conv1/Conv2DConv2D"model/block2_pool/MaxPool:output:00model/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model/block3_conv1/Conv2D?
)model/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block3_conv1/BiasAdd/ReadVariableOp?
model/block3_conv1/BiasAddBiasAdd"model/block3_conv1/Conv2D:output:01model/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
model/block3_conv1/BiasAdd?
model/block3_conv1/ReluRelu#model/block3_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
model/block3_conv1/Relu?
(model/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block3_conv2/Conv2D/ReadVariableOp?
model/block3_conv2/Conv2DConv2D%model/block3_conv1/Relu:activations:00model/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model/block3_conv2/Conv2D?
)model/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block3_conv2/BiasAdd/ReadVariableOp?
model/block3_conv2/BiasAddBiasAdd"model/block3_conv2/Conv2D:output:01model/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
model/block3_conv2/BiasAdd?
model/block3_conv2/ReluRelu#model/block3_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
model/block3_conv2/Relu?
(model/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block3_conv3/Conv2D/ReadVariableOp?
model/block3_conv3/Conv2DConv2D%model/block3_conv2/Relu:activations:00model/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model/block3_conv3/Conv2D?
)model/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block3_conv3/BiasAdd/ReadVariableOp?
model/block3_conv3/BiasAddBiasAdd"model/block3_conv3/Conv2D:output:01model/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
model/block3_conv3/BiasAdd?
model/block3_conv3/ReluRelu#model/block3_conv3/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
model/block3_conv3/Relu?
(model/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block3_conv4/Conv2D/ReadVariableOp?
model/block3_conv4/Conv2DConv2D%model/block3_conv3/Relu:activations:00model/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model/block3_conv4/Conv2D?
)model/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block3_conv4/BiasAdd/ReadVariableOp?
model/block3_conv4/BiasAddBiasAdd"model/block3_conv4/Conv2D:output:01model/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
model/block3_conv4/BiasAdd?
model/block3_conv4/ReluRelu#model/block3_conv4/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
model/block3_conv4/Relu?
model/block3_pool/MaxPoolMaxPool%model/block3_conv4/Relu:activations:0*0
_output_shapes
:?????????dd?*
ksize
*
paddingVALID*
strides
2
model/block3_pool/MaxPool?
(model/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block4_conv1/Conv2D/ReadVariableOp?
model/block4_conv1/Conv2DConv2D"model/block3_pool/MaxPool:output:00model/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
model/block4_conv1/Conv2D?
)model/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block4_conv1/BiasAdd/ReadVariableOp?
model/block4_conv1/BiasAddBiasAdd"model/block4_conv1/Conv2D:output:01model/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
model/block4_conv1/BiasAdd?
model/block4_conv1/ReluRelu#model/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
model/block4_conv1/Relu?
(model/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block4_conv2/Conv2D/ReadVariableOp?
model/block4_conv2/Conv2DConv2D%model/block4_conv1/Relu:activations:00model/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
model/block4_conv2/Conv2D?
)model/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block4_conv2/BiasAdd/ReadVariableOp?
model/block4_conv2/BiasAddBiasAdd"model/block4_conv2/Conv2D:output:01model/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
model/block4_conv2/BiasAdd?
model/block4_conv2/ReluRelu#model/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
model/block4_conv2/Relu?
(model/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block4_conv3/Conv2D/ReadVariableOp?
model/block4_conv3/Conv2DConv2D%model/block4_conv2/Relu:activations:00model/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
model/block4_conv3/Conv2D?
)model/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block4_conv3/BiasAdd/ReadVariableOp?
model/block4_conv3/BiasAddBiasAdd"model/block4_conv3/Conv2D:output:01model/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
model/block4_conv3/BiasAdd?
model/block4_conv3/ReluRelu#model/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
model/block4_conv3/Relu?
(model/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block4_conv4/Conv2D/ReadVariableOp?
model/block4_conv4/Conv2DConv2D%model/block4_conv3/Relu:activations:00model/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
model/block4_conv4/Conv2D?
)model/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block4_conv4/BiasAdd/ReadVariableOp?
model/block4_conv4/BiasAddBiasAdd"model/block4_conv4/Conv2D:output:01model/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2
model/block4_conv4/BiasAdd?
model/block4_conv4/ReluRelu#model/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
model/block4_conv4/Relu?
model/block4_pool/MaxPoolMaxPool%model/block4_conv4/Relu:activations:0*0
_output_shapes
:?????????22?*
ksize
*
paddingVALID*
strides
2
model/block4_pool/MaxPool?
(model/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block5_conv1/Conv2D/ReadVariableOp?
model/block5_conv1/Conv2DConv2D"model/block4_pool/MaxPool:output:00model/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
model/block5_conv1/Conv2D?
)model/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block5_conv1/BiasAdd/ReadVariableOp?
model/block5_conv1/BiasAddBiasAdd"model/block5_conv1/Conv2D:output:01model/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
model/block5_conv1/BiasAdd?
model/block5_conv1/ReluRelu#model/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
model/block5_conv1/Relu?
(model/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block5_conv2/Conv2D/ReadVariableOp?
model/block5_conv2/Conv2DConv2D%model/block5_conv1/Relu:activations:00model/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
model/block5_conv2/Conv2D?
)model/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block5_conv2/BiasAdd/ReadVariableOp?
model/block5_conv2/BiasAddBiasAdd"model/block5_conv2/Conv2D:output:01model/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
model/block5_conv2/BiasAdd?
model/block5_conv2/ReluRelu#model/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
model/block5_conv2/Relu?
(model/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block5_conv3/Conv2D/ReadVariableOp?
model/block5_conv3/Conv2DConv2D%model/block5_conv2/Relu:activations:00model/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
model/block5_conv3/Conv2D?
)model/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block5_conv3/BiasAdd/ReadVariableOp?
model/block5_conv3/BiasAddBiasAdd"model/block5_conv3/Conv2D:output:01model/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
model/block5_conv3/BiasAdd?
model/block5_conv3/ReluRelu#model/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
model/block5_conv3/Relu?
(model/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model/block5_conv4/Conv2D/ReadVariableOp?
model/block5_conv4/Conv2DConv2D%model/block5_conv3/Relu:activations:00model/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?*
paddingSAME*
strides
2
model/block5_conv4/Conv2D?
)model/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model/block5_conv4/BiasAdd/ReadVariableOp?
model/block5_conv4/BiasAddBiasAdd"model/block5_conv4/Conv2D:output:01model/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????22?2
model/block5_conv4/BiasAdd?
model/block5_conv4/ReluRelu#model/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????22?2
model/block5_conv4/Relu?
model/block5_pool/MaxPoolMaxPool%model/block5_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
model/block5_pool/MaxPool?
*model/block_6_conv_1/Conv2D/ReadVariableOpReadVariableOp3model_block_6_conv_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*model/block_6_conv_1/Conv2D/ReadVariableOp?
model/block_6_conv_1/Conv2DConv2D"model/block5_pool/MaxPool:output:02model/block_6_conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
model/block_6_conv_1/Conv2D?
+model/block_6_conv_1/BiasAdd/ReadVariableOpReadVariableOp4model_block_6_conv_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+model/block_6_conv_1/BiasAdd/ReadVariableOp?
model/block_6_conv_1/BiasAddBiasAdd$model/block_6_conv_1/Conv2D:output:03model/block_6_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/block_6_conv_1/BiasAdd?
model/block_6_conv_1/ReluRelu%model/block_6_conv_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/block_6_conv_1/Relu?
model/dropout/IdentityIdentity'model/block_6_conv_1/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model/dropout/Identity?
*model/block_6_conv_2/Conv2D/ReadVariableOpReadVariableOp3model_block_6_conv_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*model/block_6_conv_2/Conv2D/ReadVariableOp?
model/block_6_conv_2/Conv2DConv2Dmodel/dropout/Identity:output:02model/block_6_conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
model/block_6_conv_2/Conv2D?
+model/block_6_conv_2/BiasAdd/ReadVariableOpReadVariableOp4model_block_6_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+model/block_6_conv_2/BiasAdd/ReadVariableOp?
model/block_6_conv_2/BiasAddBiasAdd$model/block_6_conv_2/Conv2D:output:03model/block_6_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/block_6_conv_2/BiasAdd?
model/block_6_conv_2/ReluRelu%model/block_6_conv_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/block_6_conv_2/Relu?
model/dropout_1/IdentityIdentity'model/block_6_conv_2/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model/dropout_1/Identity?
2model/table_decoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;model_table_decoder_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype024
2model/table_decoder/conv2d_1/Conv2D/ReadVariableOp?
#model/table_decoder/conv2d_1/Conv2DConv2D!model/dropout_1/Identity:output:0:model/table_decoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2%
#model/table_decoder/conv2d_1/Conv2D?
3model/table_decoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<model_table_decoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype025
3model/table_decoder/conv2d_1/BiasAdd/ReadVariableOp?
$model/table_decoder/conv2d_1/BiasAddBiasAdd,model/table_decoder/conv2d_1/Conv2D:output:0;model/table_decoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2&
$model/table_decoder/conv2d_1/BiasAdd?
!model/table_decoder/conv2d_1/ReluRelu-model/table_decoder/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2#
!model/table_decoder/conv2d_1/Relu?
&model/table_decoder/dropout_2/IdentityIdentity/model/table_decoder/conv2d_1/Relu:activations:0*
T0*0
_output_shapes
:??????????2(
&model/table_decoder/dropout_2/Identity?
2model/table_decoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;model_table_decoder_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype024
2model/table_decoder/conv2d_2/Conv2D/ReadVariableOp?
#model/table_decoder/conv2d_2/Conv2DConv2D/model/table_decoder/dropout_2/Identity:output:0:model/table_decoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2%
#model/table_decoder/conv2d_2/Conv2D?
3model/table_decoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp<model_table_decoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype025
3model/table_decoder/conv2d_2/BiasAdd/ReadVariableOp?
$model/table_decoder/conv2d_2/BiasAddBiasAdd,model/table_decoder/conv2d_2/Conv2D:output:0;model/table_decoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2&
$model/table_decoder/conv2d_2/BiasAdd?
!model/table_decoder/conv2d_2/ReluRelu-model/table_decoder/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2#
!model/table_decoder/conv2d_2/Relu?
)model/table_decoder/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2+
)model/table_decoder/up_sampling2d_4/Const?
+model/table_decoder/up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2-
+model/table_decoder/up_sampling2d_4/Const_1?
'model/table_decoder/up_sampling2d_4/mulMul2model/table_decoder/up_sampling2d_4/Const:output:04model/table_decoder/up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2)
'model/table_decoder/up_sampling2d_4/mul?
9model/table_decoder/up_sampling2d_4/resize/ResizeBilinearResizeBilinear/model/table_decoder/conv2d_1/Relu:activations:0+model/table_decoder/up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(2;
9model/table_decoder/up_sampling2d_4/resize/ResizeBilinear?
+model/table_decoder/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+model/table_decoder/concatenate/concat/axis?
&model/table_decoder/concatenate/concatConcatV2"model/block4_pool/MaxPool:output:0Jmodel/table_decoder/up_sampling2d_4/resize/ResizeBilinear:resized_images:04model/table_decoder/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2(
&model/table_decoder/concatenate/concat?
)model/table_decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2+
)model/table_decoder/up_sampling2d_5/Const?
+model/table_decoder/up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2-
+model/table_decoder/up_sampling2d_5/Const_1?
'model/table_decoder/up_sampling2d_5/mulMul2model/table_decoder/up_sampling2d_5/Const:output:04model/table_decoder/up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:2)
'model/table_decoder/up_sampling2d_5/mul?
9model/table_decoder/up_sampling2d_5/resize/ResizeBilinearResizeBilinear/model/table_decoder/concatenate/concat:output:0+model/table_decoder/up_sampling2d_5/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(2;
9model/table_decoder/up_sampling2d_5/resize/ResizeBilinear?
-model/table_decoder/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-model/table_decoder/concatenate_1/concat/axis?
(model/table_decoder/concatenate_1/concatConcatV2"model/block3_pool/MaxPool:output:0Jmodel/table_decoder/up_sampling2d_5/resize/ResizeBilinear:resized_images:06model/table_decoder/concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2*
(model/table_decoder/concatenate_1/concat?
)model/table_decoder/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2+
)model/table_decoder/up_sampling2d_6/Const?
+model/table_decoder/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2-
+model/table_decoder/up_sampling2d_6/Const_1?
'model/table_decoder/up_sampling2d_6/mulMul2model/table_decoder/up_sampling2d_6/Const:output:04model/table_decoder/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2)
'model/table_decoder/up_sampling2d_6/mul?
@model/table_decoder/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor1model/table_decoder/concatenate_1/concat:output:0+model/table_decoder/up_sampling2d_6/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2B
@model/table_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor?
)model/table_decoder/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2+
)model/table_decoder/up_sampling2d_7/Const?
+model/table_decoder/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2-
+model/table_decoder/up_sampling2d_7/Const_1?
'model/table_decoder/up_sampling2d_7/mulMul2model/table_decoder/up_sampling2d_7/Const:output:04model/table_decoder/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2)
'model/table_decoder/up_sampling2d_7/mul?
@model/table_decoder/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborQmodel/table_decoder/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0+model/table_decoder/up_sampling2d_7/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2B
@model/table_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor?
,model/table_decoder/conv2d_transpose_1/ShapeShapeQmodel/table_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2.
,model/table_decoder/conv2d_transpose_1/Shape?
:model/table_decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:model/table_decoder/conv2d_transpose_1/strided_slice/stack?
<model/table_decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<model/table_decoder/conv2d_transpose_1/strided_slice/stack_1?
<model/table_decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model/table_decoder/conv2d_transpose_1/strided_slice/stack_2?
4model/table_decoder/conv2d_transpose_1/strided_sliceStridedSlice5model/table_decoder/conv2d_transpose_1/Shape:output:0Cmodel/table_decoder/conv2d_transpose_1/strided_slice/stack:output:0Emodel/table_decoder/conv2d_transpose_1/strided_slice/stack_1:output:0Emodel/table_decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model/table_decoder/conv2d_transpose_1/strided_slice?
.model/table_decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?20
.model/table_decoder/conv2d_transpose_1/stack/1?
.model/table_decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?20
.model/table_decoder/conv2d_transpose_1/stack/2?
.model/table_decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :20
.model/table_decoder/conv2d_transpose_1/stack/3?
,model/table_decoder/conv2d_transpose_1/stackPack=model/table_decoder/conv2d_transpose_1/strided_slice:output:07model/table_decoder/conv2d_transpose_1/stack/1:output:07model/table_decoder/conv2d_transpose_1/stack/2:output:07model/table_decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2.
,model/table_decoder/conv2d_transpose_1/stack?
<model/table_decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<model/table_decoder/conv2d_transpose_1/strided_slice_1/stack?
>model/table_decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>model/table_decoder/conv2d_transpose_1/strided_slice_1/stack_1?
>model/table_decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>model/table_decoder/conv2d_transpose_1/strided_slice_1/stack_2?
6model/table_decoder/conv2d_transpose_1/strided_slice_1StridedSlice5model/table_decoder/conv2d_transpose_1/stack:output:0Emodel/table_decoder/conv2d_transpose_1/strided_slice_1/stack:output:0Gmodel/table_decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0Gmodel/table_decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6model/table_decoder/conv2d_transpose_1/strided_slice_1?
Fmodel/table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpOmodel_table_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02H
Fmodel/table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
7model/table_decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput5model/table_decoder/conv2d_transpose_1/stack:output:0Nmodel/table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Qmodel/table_decoder/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
29
7model/table_decoder/conv2d_transpose_1/conv2d_transpose?
=model/table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpFmodel_table_decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=model/table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?
.model/table_decoder/conv2d_transpose_1/BiasAddBiasAdd@model/table_decoder/conv2d_transpose_1/conv2d_transpose:output:0Emodel/table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????20
.model/table_decoder/conv2d_transpose_1/BiasAdd?
.model/table_decoder/conv2d_transpose_1/SoftmaxSoftmax7model/table_decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????20
.model/table_decoder/conv2d_transpose_1/Softmax?
.model/col_decoder/conv2d/Conv2D/ReadVariableOpReadVariableOp7model_col_decoder_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype020
.model/col_decoder/conv2d/Conv2D/ReadVariableOp?
model/col_decoder/conv2d/Conv2DConv2D!model/dropout_1/Identity:output:06model/col_decoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2!
model/col_decoder/conv2d/Conv2D?
/model/col_decoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp8model_col_decoder_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model/col_decoder/conv2d/BiasAdd/ReadVariableOp?
 model/col_decoder/conv2d/BiasAddBiasAdd(model/col_decoder/conv2d/Conv2D:output:07model/col_decoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 model/col_decoder/conv2d/BiasAdd?
model/col_decoder/conv2d/ReluRelu)model/col_decoder/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/col_decoder/conv2d/Relu?
%model/col_decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model/col_decoder/up_sampling2d/Const?
'model/col_decoder/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2)
'model/col_decoder/up_sampling2d/Const_1?
#model/col_decoder/up_sampling2d/mulMul.model/col_decoder/up_sampling2d/Const:output:00model/col_decoder/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2%
#model/col_decoder/up_sampling2d/mul?
5model/col_decoder/up_sampling2d/resize/ResizeBilinearResizeBilinear+model/col_decoder/conv2d/Relu:activations:0'model/col_decoder/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(27
5model/col_decoder/up_sampling2d/resize/ResizeBilinear?
+model/col_decoder/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+model/col_decoder/concatenate_2/concat/axis?
&model/col_decoder/concatenate_2/concatConcatV2"model/block4_pool/MaxPool:output:0Fmodel/col_decoder/up_sampling2d/resize/ResizeBilinear:resized_images:04model/col_decoder/concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2(
&model/col_decoder/concatenate_2/concat?
'model/col_decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2)
'model/col_decoder/up_sampling2d_1/Const?
)model/col_decoder/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)model/col_decoder/up_sampling2d_1/Const_1?
%model/col_decoder/up_sampling2d_1/mulMul0model/col_decoder/up_sampling2d_1/Const:output:02model/col_decoder/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2'
%model/col_decoder/up_sampling2d_1/mul?
7model/col_decoder/up_sampling2d_1/resize/ResizeBilinearResizeBilinear/model/col_decoder/concatenate_2/concat:output:0)model/col_decoder/up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(29
7model/col_decoder/up_sampling2d_1/resize/ResizeBilinear?
+model/col_decoder/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+model/col_decoder/concatenate_3/concat/axis?
&model/col_decoder/concatenate_3/concatConcatV2"model/block3_pool/MaxPool:output:0Hmodel/col_decoder/up_sampling2d_1/resize/ResizeBilinear:resized_images:04model/col_decoder/concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2(
&model/col_decoder/concatenate_3/concat?
'model/col_decoder/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2)
'model/col_decoder/up_sampling2d_2/Const?
)model/col_decoder/up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)model/col_decoder/up_sampling2d_2/Const_1?
%model/col_decoder/up_sampling2d_2/mulMul0model/col_decoder/up_sampling2d_2/Const:output:02model/col_decoder/up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:2'
%model/col_decoder/up_sampling2d_2/mul?
>model/col_decoder/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor/model/col_decoder/concatenate_3/concat:output:0)model/col_decoder/up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2@
>model/col_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor?
'model/col_decoder/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2)
'model/col_decoder/up_sampling2d_3/Const?
)model/col_decoder/up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)model/col_decoder/up_sampling2d_3/Const_1?
%model/col_decoder/up_sampling2d_3/mulMul0model/col_decoder/up_sampling2d_3/Const:output:02model/col_decoder/up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:2'
%model/col_decoder/up_sampling2d_3/mul?
>model/col_decoder/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborOmodel/col_decoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0)model/col_decoder/up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2@
>model/col_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor?
(model/col_decoder/conv2d_transpose/ShapeShapeOmodel/col_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2*
(model/col_decoder/conv2d_transpose/Shape?
6model/col_decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6model/col_decoder/conv2d_transpose/strided_slice/stack?
8model/col_decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/col_decoder/conv2d_transpose/strided_slice/stack_1?
8model/col_decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/col_decoder/conv2d_transpose/strided_slice/stack_2?
0model/col_decoder/conv2d_transpose/strided_sliceStridedSlice1model/col_decoder/conv2d_transpose/Shape:output:0?model/col_decoder/conv2d_transpose/strided_slice/stack:output:0Amodel/col_decoder/conv2d_transpose/strided_slice/stack_1:output:0Amodel/col_decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0model/col_decoder/conv2d_transpose/strided_slice?
*model/col_decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2,
*model/col_decoder/conv2d_transpose/stack/1?
*model/col_decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2,
*model/col_decoder/conv2d_transpose/stack/2?
*model/col_decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2,
*model/col_decoder/conv2d_transpose/stack/3?
(model/col_decoder/conv2d_transpose/stackPack9model/col_decoder/conv2d_transpose/strided_slice:output:03model/col_decoder/conv2d_transpose/stack/1:output:03model/col_decoder/conv2d_transpose/stack/2:output:03model/col_decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2*
(model/col_decoder/conv2d_transpose/stack?
8model/col_decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model/col_decoder/conv2d_transpose/strided_slice_1/stack?
:model/col_decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model/col_decoder/conv2d_transpose/strided_slice_1/stack_1?
:model/col_decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model/col_decoder/conv2d_transpose/strided_slice_1/stack_2?
2model/col_decoder/conv2d_transpose/strided_slice_1StridedSlice1model/col_decoder/conv2d_transpose/stack:output:0Amodel/col_decoder/conv2d_transpose/strided_slice_1/stack:output:0Cmodel/col_decoder/conv2d_transpose/strided_slice_1/stack_1:output:0Cmodel/col_decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2model/col_decoder/conv2d_transpose/strided_slice_1?
Bmodel/col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpKmodel_col_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02D
Bmodel/col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?
3model/col_decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput1model/col_decoder/conv2d_transpose/stack:output:0Jmodel/col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0Omodel/col_decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
25
3model/col_decoder/conv2d_transpose/conv2d_transpose?
9model/col_decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpBmodel_col_decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9model/col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp?
*model/col_decoder/conv2d_transpose/BiasAddBiasAdd<model/col_decoder/conv2d_transpose/conv2d_transpose:output:0Amodel/col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2,
*model/col_decoder/conv2d_transpose/BiasAdd?
*model/col_decoder/conv2d_transpose/SoftmaxSoftmax3model/col_decoder/conv2d_transpose/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2,
*model/col_decoder/conv2d_transpose/Softmax?
IdentityIdentity4model/col_decoder/conv2d_transpose/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity8model/table_decoder/conv2d_transpose_1/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?
NoOpNoOp*^model/block1_conv1/BiasAdd/ReadVariableOp)^model/block1_conv1/Conv2D/ReadVariableOp*^model/block1_conv2/BiasAdd/ReadVariableOp)^model/block1_conv2/Conv2D/ReadVariableOp*^model/block2_conv1/BiasAdd/ReadVariableOp)^model/block2_conv1/Conv2D/ReadVariableOp*^model/block2_conv2/BiasAdd/ReadVariableOp)^model/block2_conv2/Conv2D/ReadVariableOp*^model/block3_conv1/BiasAdd/ReadVariableOp)^model/block3_conv1/Conv2D/ReadVariableOp*^model/block3_conv2/BiasAdd/ReadVariableOp)^model/block3_conv2/Conv2D/ReadVariableOp*^model/block3_conv3/BiasAdd/ReadVariableOp)^model/block3_conv3/Conv2D/ReadVariableOp*^model/block3_conv4/BiasAdd/ReadVariableOp)^model/block3_conv4/Conv2D/ReadVariableOp*^model/block4_conv1/BiasAdd/ReadVariableOp)^model/block4_conv1/Conv2D/ReadVariableOp*^model/block4_conv2/BiasAdd/ReadVariableOp)^model/block4_conv2/Conv2D/ReadVariableOp*^model/block4_conv3/BiasAdd/ReadVariableOp)^model/block4_conv3/Conv2D/ReadVariableOp*^model/block4_conv4/BiasAdd/ReadVariableOp)^model/block4_conv4/Conv2D/ReadVariableOp*^model/block5_conv1/BiasAdd/ReadVariableOp)^model/block5_conv1/Conv2D/ReadVariableOp*^model/block5_conv2/BiasAdd/ReadVariableOp)^model/block5_conv2/Conv2D/ReadVariableOp*^model/block5_conv3/BiasAdd/ReadVariableOp)^model/block5_conv3/Conv2D/ReadVariableOp*^model/block5_conv4/BiasAdd/ReadVariableOp)^model/block5_conv4/Conv2D/ReadVariableOp,^model/block_6_conv_1/BiasAdd/ReadVariableOp+^model/block_6_conv_1/Conv2D/ReadVariableOp,^model/block_6_conv_2/BiasAdd/ReadVariableOp+^model/block_6_conv_2/Conv2D/ReadVariableOp0^model/col_decoder/conv2d/BiasAdd/ReadVariableOp/^model/col_decoder/conv2d/Conv2D/ReadVariableOp:^model/col_decoder/conv2d_transpose/BiasAdd/ReadVariableOpC^model/col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp4^model/table_decoder/conv2d_1/BiasAdd/ReadVariableOp3^model/table_decoder/conv2d_1/Conv2D/ReadVariableOp4^model/table_decoder/conv2d_2/BiasAdd/ReadVariableOp3^model/table_decoder/conv2d_2/Conv2D/ReadVariableOp>^model/table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpG^model/table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)model/block1_conv1/BiasAdd/ReadVariableOp)model/block1_conv1/BiasAdd/ReadVariableOp2T
(model/block1_conv1/Conv2D/ReadVariableOp(model/block1_conv1/Conv2D/ReadVariableOp2V
)model/block1_conv2/BiasAdd/ReadVariableOp)model/block1_conv2/BiasAdd/ReadVariableOp2T
(model/block1_conv2/Conv2D/ReadVariableOp(model/block1_conv2/Conv2D/ReadVariableOp2V
)model/block2_conv1/BiasAdd/ReadVariableOp)model/block2_conv1/BiasAdd/ReadVariableOp2T
(model/block2_conv1/Conv2D/ReadVariableOp(model/block2_conv1/Conv2D/ReadVariableOp2V
)model/block2_conv2/BiasAdd/ReadVariableOp)model/block2_conv2/BiasAdd/ReadVariableOp2T
(model/block2_conv2/Conv2D/ReadVariableOp(model/block2_conv2/Conv2D/ReadVariableOp2V
)model/block3_conv1/BiasAdd/ReadVariableOp)model/block3_conv1/BiasAdd/ReadVariableOp2T
(model/block3_conv1/Conv2D/ReadVariableOp(model/block3_conv1/Conv2D/ReadVariableOp2V
)model/block3_conv2/BiasAdd/ReadVariableOp)model/block3_conv2/BiasAdd/ReadVariableOp2T
(model/block3_conv2/Conv2D/ReadVariableOp(model/block3_conv2/Conv2D/ReadVariableOp2V
)model/block3_conv3/BiasAdd/ReadVariableOp)model/block3_conv3/BiasAdd/ReadVariableOp2T
(model/block3_conv3/Conv2D/ReadVariableOp(model/block3_conv3/Conv2D/ReadVariableOp2V
)model/block3_conv4/BiasAdd/ReadVariableOp)model/block3_conv4/BiasAdd/ReadVariableOp2T
(model/block3_conv4/Conv2D/ReadVariableOp(model/block3_conv4/Conv2D/ReadVariableOp2V
)model/block4_conv1/BiasAdd/ReadVariableOp)model/block4_conv1/BiasAdd/ReadVariableOp2T
(model/block4_conv1/Conv2D/ReadVariableOp(model/block4_conv1/Conv2D/ReadVariableOp2V
)model/block4_conv2/BiasAdd/ReadVariableOp)model/block4_conv2/BiasAdd/ReadVariableOp2T
(model/block4_conv2/Conv2D/ReadVariableOp(model/block4_conv2/Conv2D/ReadVariableOp2V
)model/block4_conv3/BiasAdd/ReadVariableOp)model/block4_conv3/BiasAdd/ReadVariableOp2T
(model/block4_conv3/Conv2D/ReadVariableOp(model/block4_conv3/Conv2D/ReadVariableOp2V
)model/block4_conv4/BiasAdd/ReadVariableOp)model/block4_conv4/BiasAdd/ReadVariableOp2T
(model/block4_conv4/Conv2D/ReadVariableOp(model/block4_conv4/Conv2D/ReadVariableOp2V
)model/block5_conv1/BiasAdd/ReadVariableOp)model/block5_conv1/BiasAdd/ReadVariableOp2T
(model/block5_conv1/Conv2D/ReadVariableOp(model/block5_conv1/Conv2D/ReadVariableOp2V
)model/block5_conv2/BiasAdd/ReadVariableOp)model/block5_conv2/BiasAdd/ReadVariableOp2T
(model/block5_conv2/Conv2D/ReadVariableOp(model/block5_conv2/Conv2D/ReadVariableOp2V
)model/block5_conv3/BiasAdd/ReadVariableOp)model/block5_conv3/BiasAdd/ReadVariableOp2T
(model/block5_conv3/Conv2D/ReadVariableOp(model/block5_conv3/Conv2D/ReadVariableOp2V
)model/block5_conv4/BiasAdd/ReadVariableOp)model/block5_conv4/BiasAdd/ReadVariableOp2T
(model/block5_conv4/Conv2D/ReadVariableOp(model/block5_conv4/Conv2D/ReadVariableOp2Z
+model/block_6_conv_1/BiasAdd/ReadVariableOp+model/block_6_conv_1/BiasAdd/ReadVariableOp2X
*model/block_6_conv_1/Conv2D/ReadVariableOp*model/block_6_conv_1/Conv2D/ReadVariableOp2Z
+model/block_6_conv_2/BiasAdd/ReadVariableOp+model/block_6_conv_2/BiasAdd/ReadVariableOp2X
*model/block_6_conv_2/Conv2D/ReadVariableOp*model/block_6_conv_2/Conv2D/ReadVariableOp2b
/model/col_decoder/conv2d/BiasAdd/ReadVariableOp/model/col_decoder/conv2d/BiasAdd/ReadVariableOp2`
.model/col_decoder/conv2d/Conv2D/ReadVariableOp.model/col_decoder/conv2d/Conv2D/ReadVariableOp2v
9model/col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp9model/col_decoder/conv2d_transpose/BiasAdd/ReadVariableOp2?
Bmodel/col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpBmodel/col_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2j
3model/table_decoder/conv2d_1/BiasAdd/ReadVariableOp3model/table_decoder/conv2d_1/BiasAdd/ReadVariableOp2h
2model/table_decoder/conv2d_1/Conv2D/ReadVariableOp2model/table_decoder/conv2d_1/Conv2D/ReadVariableOp2j
3model/table_decoder/conv2d_2/BiasAdd/ReadVariableOp3model/table_decoder/conv2d_2/BiasAdd/ReadVariableOp2h
2model/table_decoder/conv2d_2/Conv2D/ReadVariableOp2model/table_decoder/conv2d_2/Conv2D/ReadVariableOp2~
=model/table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp=model/table_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
Fmodel/table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpFmodel/table_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameInput_Layer
?
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_4234

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_up_sampling2d_7_layer_call_fn_5117

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_16442
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block1_conv1_layer_call_and_return_conditional_losses_1770

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_block3_pool_layer_call_fn_4409

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_19112
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????dd?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_1275

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_3389
input_layer+
block1_conv1_3268:@
block1_conv1_3270:@+
block1_conv2_3273:@@
block1_conv2_3275:@,
block2_conv1_3279:@? 
block2_conv1_3281:	?-
block2_conv2_3284:?? 
block2_conv2_3286:	?-
block3_conv1_3290:?? 
block3_conv1_3292:	?-
block3_conv2_3295:?? 
block3_conv2_3297:	?-
block3_conv3_3300:?? 
block3_conv3_3302:	?-
block3_conv4_3305:?? 
block3_conv4_3307:	?-
block4_conv1_3311:?? 
block4_conv1_3313:	?-
block4_conv2_3316:?? 
block4_conv2_3318:	?-
block4_conv3_3321:?? 
block4_conv3_3323:	?-
block4_conv4_3326:?? 
block4_conv4_3328:	?-
block5_conv1_3332:?? 
block5_conv1_3334:	?-
block5_conv2_3337:?? 
block5_conv2_3339:	?-
block5_conv3_3342:?? 
block5_conv3_3344:	?-
block5_conv4_3347:?? 
block5_conv4_3349:	?/
block_6_conv_1_3353:??"
block_6_conv_1_3355:	?/
block_6_conv_2_3359:??"
block_6_conv_2_3361:	?.
table_decoder_3365:??!
table_decoder_3367:	?.
table_decoder_3369:??!
table_decoder_3371:	?-
table_decoder_3373:? 
table_decoder_3375:,
col_decoder_3378:??
col_decoder_3380:	?+
col_decoder_3382:?
col_decoder_3384:
identity

identity_1??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block3_conv4/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block4_conv4/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?$block5_conv4/StatefulPartitionedCall?&block_6_conv_1/StatefulPartitionedCall?&block_6_conv_2/StatefulPartitionedCall?#col_decoder/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?%table_decoder/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_layerblock1_conv1_3268block1_conv1_3270*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_17702&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_3273block1_conv2_3275*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_17872&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_17972
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_3279block2_conv1_3281*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_18102&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_3284block2_conv2_3286*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_18272&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_18372
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_3290block3_conv1_3292*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_18502&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_3295block3_conv2_3297*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_18672&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_3300block3_conv3_3302*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_18842&
$block3_conv3/StatefulPartitionedCall?
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_3305block3_conv4_3307*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_19012&
$block3_conv4/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_19112
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_3311block4_conv1_3313*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_19242&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_3316block4_conv2_3318*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_19412&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_3321block4_conv3_3323*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_19582&
$block4_conv3/StatefulPartitionedCall?
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_3326block4_conv4_3328*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_19752&
$block4_conv4/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_19852
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_3332block5_conv1_3334*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_19982&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_3337block5_conv2_3339*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_20152&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_3342block5_conv3_3344*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_20322&
$block5_conv3/StatefulPartitionedCall?
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_3347block5_conv4_3349*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_20492&
$block5_conv4/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_20592
block5_pool/PartitionedCall?
&block_6_conv_1/StatefulPartitionedCallStatefulPartitionedCall$block5_pool/PartitionedCall:output:0block_6_conv_1_3353block_6_conv_1_3355*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_20722(
&block_6_conv_1/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall/block_6_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_25222!
dropout/StatefulPartitionedCall?
&block_6_conv_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0block_6_conv_2_3359block_6_conv_2_3361*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_20962(
&block_6_conv_2/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall/block_6_conv_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_24892#
!dropout_1/StatefulPartitionedCall?
%table_decoder/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0table_decoder_3365table_decoder_3367table_decoder_3369table_decoder_3371table_decoder_3373table_decoder_3375*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_24542'
%table_decoder/StatefulPartitionedCall?
#col_decoder/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0col_decoder_3378col_decoder_3380col_decoder_3382col_decoder_3384*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_col_decoder_layer_call_and_return_conditional_losses_22372%
#col_decoder/StatefulPartitionedCall?
IdentityIdentity,col_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity.table_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2P
&block_6_conv_1/StatefulPartitionedCall&block_6_conv_1/StatefulPartitionedCall2P
&block_6_conv_2/StatefulPartitionedCall&block_6_conv_2/StatefulPartitionedCall2J
#col_decoder/StatefulPartitionedCall#col_decoder/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2N
%table_decoder/StatefulPartitionedCall%table_decoder/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameInput_Layer
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_2522

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv1_layer_call_and_return_conditional_losses_4420

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
F
*__inference_block1_pool_layer_call_fn_4249

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_17972
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_4090

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?&

unknown_33:??

unknown_34:	?&

unknown_35:??

unknown_36:	?&

unknown_37:??

unknown_38:	?%

unknown_39:?

unknown_40:&

unknown_41:??

unknown_42:	?%

unknown_43:?

unknown_44:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::???????????:???????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_22492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_block_6_conv_2_layer_call_fn_4676

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_20962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_4189

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?&

unknown_33:??

unknown_34:	?&

unknown_35:??

unknown_36:	?&

unknown_37:??

unknown_38:	?%

unknown_39:?

unknown_40:&

unknown_41:??

unknown_42:	?%

unknown_43:?

unknown_44:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::???????????:???????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_29452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_4984

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block3_conv1_layer_call_and_return_conditional_losses_1850

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv3_layer_call_and_return_conditional_losses_1958

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_3141
input_layer!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?&

unknown_33:??

unknown_34:	?&

unknown_35:??

unknown_36:	?&

unknown_37:??

unknown_38:	?%

unknown_39:?

unknown_40:&

unknown_41:??

unknown_42:	?%

unknown_43:?

unknown_44:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::???????????:???????????*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_29452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameInput_Layer
?
J
.__inference_up_sampling2d_4_layer_call_fn_5066

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_15362
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?I
?
E__inference_col_decoder_layer_call_and_return_conditional_losses_4756
input_0
input_1
input_2A
%conv2d_conv2d_readvariableop_resource:??5
&conv2d_biasadd_readvariableop_resource:	?T
9conv2d_transpose_conv2d_transpose_readvariableop_resource:?>
0conv2d_transpose_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAddv
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d/Relu{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
#up_sampling2d/resize/ResizeBilinearResizeBilinearconv2d/Relu:activations:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(2%
#up_sampling2d/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2input_14up_sampling2d/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2
concatenate/concat
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
%up_sampling2d_1/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(2'
%up_sampling2d_1/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2input_26up_sampling2d_1/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2
concatenate_1/concat
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2
up_sampling2d_2/Const?
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const_1?
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul?
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
up_sampling2d_3/Const?
up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const_1?
up_sampling2d_3/mulMulup_sampling2d_3/Const:output:0 up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul?
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor?
conv2d_transpose/ShapeShape=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/BiasAdd?
conv2d_transpose/SoftmaxSoftmax!conv2d_transpose/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/Softmax?
IdentityIdentity"conv2d_transpose/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:??????????:?????????22?:?????????dd?: : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input/0:YU
0
_output_shapes
:?????????22?
!
_user_specified_name	input/1:YU
0
_output_shapes
:?????????dd?
!
_user_specified_name	input/2
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_4634

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_4494

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_4299

inputs
identity?
MaxPoolMaxPoolinputs*2
_output_shapes 
:????????????*
ksize
*
paddingVALID*
strides
2	
MaxPoolo
IdentityIdentityMaxPool:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_3265
input_layer+
block1_conv1_3144:@
block1_conv1_3146:@+
block1_conv2_3149:@@
block1_conv2_3151:@,
block2_conv1_3155:@? 
block2_conv1_3157:	?-
block2_conv2_3160:?? 
block2_conv2_3162:	?-
block3_conv1_3166:?? 
block3_conv1_3168:	?-
block3_conv2_3171:?? 
block3_conv2_3173:	?-
block3_conv3_3176:?? 
block3_conv3_3178:	?-
block3_conv4_3181:?? 
block3_conv4_3183:	?-
block4_conv1_3187:?? 
block4_conv1_3189:	?-
block4_conv2_3192:?? 
block4_conv2_3194:	?-
block4_conv3_3197:?? 
block4_conv3_3199:	?-
block4_conv4_3202:?? 
block4_conv4_3204:	?-
block5_conv1_3208:?? 
block5_conv1_3210:	?-
block5_conv2_3213:?? 
block5_conv2_3215:	?-
block5_conv3_3218:?? 
block5_conv3_3220:	?-
block5_conv4_3223:?? 
block5_conv4_3225:	?/
block_6_conv_1_3229:??"
block_6_conv_1_3231:	?/
block_6_conv_2_3235:??"
block_6_conv_2_3237:	?.
table_decoder_3241:??!
table_decoder_3243:	?.
table_decoder_3245:??!
table_decoder_3247:	?-
table_decoder_3249:? 
table_decoder_3251:,
col_decoder_3254:??
col_decoder_3256:	?+
col_decoder_3258:?
col_decoder_3260:
identity

identity_1??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block3_conv4/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block4_conv4/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?$block5_conv4/StatefulPartitionedCall?&block_6_conv_1/StatefulPartitionedCall?&block_6_conv_2/StatefulPartitionedCall?#col_decoder/StatefulPartitionedCall?%table_decoder/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_layerblock1_conv1_3144block1_conv1_3146*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_17702&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_3149block1_conv2_3151*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_17872&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_17972
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_3155block2_conv1_3157*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_18102&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_3160block2_conv2_3162*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_18272&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_18372
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_3166block3_conv1_3168*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_18502&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_3171block3_conv2_3173*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_18672&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_3176block3_conv3_3178*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_18842&
$block3_conv3/StatefulPartitionedCall?
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_3181block3_conv4_3183*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_19012&
$block3_conv4/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_19112
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_3187block4_conv1_3189*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_19242&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_3192block4_conv2_3194*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_19412&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_3197block4_conv3_3199*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_19582&
$block4_conv3/StatefulPartitionedCall?
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_3202block4_conv4_3204*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_19752&
$block4_conv4/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_19852
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_3208block5_conv1_3210*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_19982&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_3213block5_conv2_3215*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_20152&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_3218block5_conv3_3220*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_20322&
$block5_conv3/StatefulPartitionedCall?
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_3223block5_conv4_3225*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_20492&
$block5_conv4/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_20592
block5_pool/PartitionedCall?
&block_6_conv_1/StatefulPartitionedCallStatefulPartitionedCall$block5_pool/PartitionedCall:output:0block_6_conv_1_3229block_6_conv_1_3231*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_20722(
&block_6_conv_1/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall/block_6_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_20832
dropout/PartitionedCall?
&block_6_conv_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0block_6_conv_2_3235block_6_conv_2_3237*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_20962(
&block_6_conv_2/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall/block_6_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_21072
dropout_1/PartitionedCall?
%table_decoder/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0table_decoder_3241table_decoder_3243table_decoder_3245table_decoder_3247table_decoder_3249table_decoder_3251*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_21702'
%table_decoder/StatefulPartitionedCall?
#col_decoder/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0col_decoder_3254col_decoder_3256col_decoder_3258col_decoder_3260*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_col_decoder_layer_call_and_return_conditional_losses_22372%
#col_decoder/StatefulPartitionedCall?
IdentityIdentity,col_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity.table_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2P
&block_6_conv_1/StatefulPartitionedCall&block_6_conv_1/StatefulPartitionedCall2P
&block_6_conv_2/StatefulPartitionedCall&block_6_conv_2/StatefulPartitionedCall2J
#col_decoder/StatefulPartitionedCall#col_decoder/StatefulPartitionedCall2N
%table_decoder/StatefulPartitionedCall%table_decoder/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameInput_Layer
?
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_4399

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????dd?*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????dd?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
F
*__inference_block5_pool_layer_call_fn_4604

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_12752
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_col_decoder_layer_call_fn_4771
input_0
input_1
input_2#
unknown:??
	unknown_0:	?$
	unknown_1:?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1input_2unknown	unknown_0	unknown_1	unknown_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_col_decoder_layer_call_and_return_conditional_losses_22372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:??????????:?????????22?:?????????dd?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input/0:YU
0
_output_shapes
:?????????22?
!
_user_specified_name	input/1:YU
0
_output_shapes
:?????????dd?
!
_user_specified_name	input/2
?
e
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_5078

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_2059

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????22?:X T
0
_output_shapes
:?????????22?
 
_user_specified_nameinputs
?
?
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_2072

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_2945

inputs+
block1_conv1_2824:@
block1_conv1_2826:@+
block1_conv2_2829:@@
block1_conv2_2831:@,
block2_conv1_2835:@? 
block2_conv1_2837:	?-
block2_conv2_2840:?? 
block2_conv2_2842:	?-
block3_conv1_2846:?? 
block3_conv1_2848:	?-
block3_conv2_2851:?? 
block3_conv2_2853:	?-
block3_conv3_2856:?? 
block3_conv3_2858:	?-
block3_conv4_2861:?? 
block3_conv4_2863:	?-
block4_conv1_2867:?? 
block4_conv1_2869:	?-
block4_conv2_2872:?? 
block4_conv2_2874:	?-
block4_conv3_2877:?? 
block4_conv3_2879:	?-
block4_conv4_2882:?? 
block4_conv4_2884:	?-
block5_conv1_2888:?? 
block5_conv1_2890:	?-
block5_conv2_2893:?? 
block5_conv2_2895:	?-
block5_conv3_2898:?? 
block5_conv3_2900:	?-
block5_conv4_2903:?? 
block5_conv4_2905:	?/
block_6_conv_1_2909:??"
block_6_conv_1_2911:	?/
block_6_conv_2_2915:??"
block_6_conv_2_2917:	?.
table_decoder_2921:??!
table_decoder_2923:	?.
table_decoder_2925:??!
table_decoder_2927:	?-
table_decoder_2929:? 
table_decoder_2931:,
col_decoder_2934:??
col_decoder_2936:	?+
col_decoder_2938:?
col_decoder_2940:
identity

identity_1??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block3_conv4/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block4_conv4/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?$block5_conv4/StatefulPartitionedCall?&block_6_conv_1/StatefulPartitionedCall?&block_6_conv_2/StatefulPartitionedCall?#col_decoder/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?%table_decoder/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_2824block1_conv1_2826*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_17702&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_2829block1_conv2_2831*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_17872&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_17972
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_2835block2_conv1_2837*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_18102&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_2840block2_conv2_2842*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_18272&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_18372
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_2846block3_conv1_2848*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_18502&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_2851block3_conv2_2853*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_18672&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_2856block3_conv3_2858*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_18842&
$block3_conv3/StatefulPartitionedCall?
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_2861block3_conv4_2863*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_19012&
$block3_conv4/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_19112
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_2867block4_conv1_2869*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_19242&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_2872block4_conv2_2874*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_19412&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_2877block4_conv3_2879*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_19582&
$block4_conv3/StatefulPartitionedCall?
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_2882block4_conv4_2884*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????dd?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_19752&
$block4_conv4/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_19852
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_2888block5_conv1_2890*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_19982&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_2893block5_conv2_2895*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_20152&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_2898block5_conv3_2900*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_20322&
$block5_conv3/StatefulPartitionedCall?
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_2903block5_conv4_2905*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????22?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_20492&
$block5_conv4/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_20592
block5_pool/PartitionedCall?
&block_6_conv_1/StatefulPartitionedCallStatefulPartitionedCall$block5_pool/PartitionedCall:output:0block_6_conv_1_2909block_6_conv_1_2911*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_20722(
&block_6_conv_1/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall/block_6_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_25222!
dropout/StatefulPartitionedCall?
&block_6_conv_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0block_6_conv_2_2915block_6_conv_2_2917*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_20962(
&block_6_conv_2/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall/block_6_conv_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_24892#
!dropout_1/StatefulPartitionedCall?
%table_decoder/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0table_decoder_2921table_decoder_2923table_decoder_2925table_decoder_2927table_decoder_2929table_decoder_2931*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_24542'
%table_decoder/StatefulPartitionedCall?
#col_decoder/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$block4_pool/PartitionedCall:output:0$block3_pool/PartitionedCall:output:0col_decoder_2934col_decoder_2936col_decoder_2938col_decoder_2940*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_col_decoder_layer_call_and_return_conditional_losses_22372%
#col_decoder/StatefulPartitionedCall?
IdentityIdentity,col_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?

Identity_1Identity.table_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall'^block_6_conv_1/StatefulPartitionedCall'^block_6_conv_2/StatefulPartitionedCall$^col_decoder/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall&^table_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2P
&block_6_conv_1/StatefulPartitionedCall&block_6_conv_1/StatefulPartitionedCall2P
&block_6_conv_2/StatefulPartitionedCall&block_6_conv_2/StatefulPartitionedCall2J
#col_decoder/StatefulPartitionedCall#col_decoder/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2N
%table_decoder/StatefulPartitionedCall%table_decoder/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?g
?
__inference__traced_save_5340
file_prefix2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block3_conv4_kernel_read_readvariableop0
,savev2_block3_conv4_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block4_conv4_kernel_read_readvariableop0
,savev2_block4_conv4_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop2
.savev2_block5_conv4_kernel_read_readvariableop0
,savev2_block5_conv4_bias_read_readvariableop4
0savev2_block_6_conv_1_kernel_read_readvariableop2
.savev2_block_6_conv_1_bias_read_readvariableop4
0savev2_block_6_conv_2_kernel_read_readvariableop2
.savev2_block_6_conv_2_bias_read_readvariableop8
4savev2_col_decoder_conv2d_kernel_read_readvariableop6
2savev2_col_decoder_conv2d_bias_read_readvariableopB
>savev2_col_decoder_conv2d_transpose_kernel_read_readvariableop@
<savev2_col_decoder_conv2d_transpose_bias_read_readvariableop<
8savev2_table_decoder_conv2d_1_kernel_read_readvariableop:
6savev2_table_decoder_conv2d_1_bias_read_readvariableop<
8savev2_table_decoder_conv2d_2_kernel_read_readvariableop:
6savev2_table_decoder_conv2d_2_bias_read_readvariableopF
Bsavev2_table_decoder_conv2d_transpose_1_kernel_read_readvariableopD
@savev2_table_decoder_conv2d_transpose_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*?
value?B?5B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block3_conv4_kernel_read_readvariableop,savev2_block3_conv4_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block4_conv4_kernel_read_readvariableop,savev2_block4_conv4_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop.savev2_block5_conv4_kernel_read_readvariableop,savev2_block5_conv4_bias_read_readvariableop0savev2_block_6_conv_1_kernel_read_readvariableop.savev2_block_6_conv_1_bias_read_readvariableop0savev2_block_6_conv_2_kernel_read_readvariableop.savev2_block_6_conv_2_bias_read_readvariableop4savev2_col_decoder_conv2d_kernel_read_readvariableop2savev2_col_decoder_conv2d_bias_read_readvariableop>savev2_col_decoder_conv2d_transpose_kernel_read_readvariableop<savev2_col_decoder_conv2d_transpose_bias_read_readvariableop8savev2_table_decoder_conv2d_1_kernel_read_readvariableop6savev2_table_decoder_conv2d_1_bias_read_readvariableop8savev2_table_decoder_conv2d_2_kernel_read_readvariableop6savev2_table_decoder_conv2d_2_bias_read_readvariableopBsavev2_table_decoder_conv2d_transpose_1_kernel_read_readvariableop@savev2_table_decoder_conv2d_transpose_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *C
dtypes9
7252
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@@:@:@?:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:?::??:?:??:?:?:: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.	*
(
_output_shapes
:??:!


_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:! 

_output_shapes	
:?:.!*
(
_output_shapes
:??:!"

_output_shapes	
:?:.#*
(
_output_shapes
:??:!$

_output_shapes	
:?:.%*
(
_output_shapes
:??:!&

_output_shapes	
:?:-')
'
_output_shapes
:?: (

_output_shapes
::.)*
(
_output_shapes
:??:!*

_output_shapes	
:?:.+*
(
_output_shapes
:??:!,

_output_shapes	
:?:--)
'
_output_shapes
:?: .

_output_shapes
::/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: 
?
?
+__inference_block2_conv2_layer_call_fn_4289

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_18272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?&
?
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1702

inputsC
(conv2d_transpose_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SoftmaxSoftmaxBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block2_conv1_layer_call_and_return_conditional_losses_4260

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
+__inference_block3_conv3_layer_call_fn_4369

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_18842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
F__inference_block1_conv1_layer_call_and_return_conditional_losses_4200

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
,__inference_up_sampling2d_layer_call_fn_4955

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_13042
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_2096

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_block4_pool_layer_call_fn_4504

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_12532
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_up_sampling2d_3_layer_call_fn_5006

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_14122
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_1644

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv2_layer_call_and_return_conditional_losses_1941

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????dd?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????dd?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????dd?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????dd?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????dd?
 
_user_specified_nameinputs
?
?
/__inference_conv2d_transpose_layer_call_fn_5049

inputs"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_14702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_table_decoder_layer_call_fn_4938
input_0
input_1
input_2#
unknown:??
	unknown_0:	?%
	unknown_1:??
	unknown_2:	?$
	unknown_3:?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_table_decoder_layer_call_and_return_conditional_losses_24542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:??????????:?????????22?:?????????dd?: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input/0:YU
0
_output_shapes
:?????????22?
!
_user_specified_name	input/1:YU
0
_output_shapes
:?????????dd?
!
_user_specified_name	input/2
?
?
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_4667

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_5095

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_4681

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_block2_conv1_layer_call_fn_4269

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_18102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_1253

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?I
?
E__inference_col_decoder_layer_call_and_return_conditional_losses_2237	
input
input_1
input_2A
%conv2d_conv2d_readvariableop_resource:??5
&conv2d_biasadd_readvariableop_resource:	?T
9conv2d_transpose_conv2d_transpose_readvariableop_resource:?>
0conv2d_transpose_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAddv
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d/Relu{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
#up_sampling2d/resize/ResizeBilinearResizeBilinearconv2d/Relu:activations:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:?????????22?*
half_pixel_centers(2%
#up_sampling2d/resize/ResizeBilineart
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2input_14up_sampling2d/resize/ResizeBilinear:resized_images:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????22?2
concatenate/concat
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"2   2   2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
%up_sampling2d_1/resize/ResizeBilinearResizeBilinearconcatenate/concat:output:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:?????????dd?*
half_pixel_centers(2'
%up_sampling2d_1/resize/ResizeBilinearx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2input_26up_sampling2d_1/resize/ResizeBilinear:resized_images:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????dd?2
concatenate_1/concat
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"d   d   2
up_sampling2d_2/Const?
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const_1?
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul?
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2
up_sampling2d_3/Const?
up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const_1?
up_sampling2d_3/mulMulup_sampling2d_3/Const:output:0 up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul?
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0up_sampling2d_3/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor?
conv2d_transpose/ShapeShape=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/BiasAdd?
conv2d_transpose/SoftmaxSoftmax!conv2d_transpose/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose/Softmax?
IdentityIdentity"conv2d_transpose/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:??????????:?????????22?:?????????dd?: : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:W S
0
_output_shapes
:??????????

_user_specified_nameinput:WS
0
_output_shapes
:?????????22?

_user_specified_nameinput:WS
0
_output_shapes
:?????????dd?

_user_specified_nameinput"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
Input_Layer>
serving_default_Input_Layer:0???????????I
col_decoder:
StatefulPartitionedCall:0???????????K
table_decoder:
StatefulPartitionedCall:1???????????tensorflow/serving/predict:??
?	
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer_with_weights-15
layer-20
layer-21
layer_with_weights-16
layer-22
layer-23
layer_with_weights-17
layer-24
layer-25
layer_with_weights-18
layer-26
layer_with_weights-19
layer-27
	optimizer
loss
	variables
 trainable_variables
!regularization_losses
"	keras_api
#
signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
?

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
\regularization_losses
]	variables
^trainable_variables
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

rkernel
sbias
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

|kernel
}bias
~regularization_losses
	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

?conv1
?	upsample1
?	upsample2
?	upsample3
?	upsample4
?convtraspose
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

?conv1

?conv2

?drop1
?	upsample1
?	upsample2
?	upsample3
?	upsample4
?convtraspose
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
"
	optimizer
 "
trackable_dict_wrapper
?
$0
%1
*2
+3
44
55
:6
;7
D8
E9
J10
K11
P12
Q13
V14
W15
`16
a17
f18
g19
l20
m21
r22
s23
|24
}25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
	variables
 ?layer_regularization_losses
?layers
 trainable_variables
?layer_metrics
!regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
&regularization_losses
?metrics
'	variables
 ?layer_regularization_losses
?layers
(trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
,regularization_losses
?metrics
-	variables
 ?layer_regularization_losses
?layers
.trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0regularization_losses
?metrics
1	variables
 ?layer_regularization_losses
?layers
2trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,@?2block2_conv1/kernel
 :?2block2_conv1/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
6regularization_losses
?metrics
7	variables
 ?layer_regularization_losses
?layers
8trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block2_conv2/kernel
 :?2block2_conv2/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<regularization_losses
?metrics
=	variables
 ?layer_regularization_losses
?layers
>trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@regularization_losses
?metrics
A	variables
 ?layer_regularization_losses
?layers
Btrainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block3_conv1/kernel
 :?2block3_conv1/bias
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fregularization_losses
?metrics
G	variables
 ?layer_regularization_losses
?layers
Htrainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block3_conv2/kernel
 :?2block3_conv2/bias
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Lregularization_losses
?metrics
M	variables
 ?layer_regularization_losses
?layers
Ntrainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block3_conv3/kernel
 :?2block3_conv3/bias
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rregularization_losses
?metrics
S	variables
 ?layer_regularization_losses
?layers
Ttrainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block3_conv4/kernel
 :?2block3_conv4/bias
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xregularization_losses
?metrics
Y	variables
 ?layer_regularization_losses
?layers
Ztrainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
\regularization_losses
?metrics
]	variables
 ?layer_regularization_losses
?layers
^trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block4_conv1/kernel
 :?2block4_conv1/bias
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
bregularization_losses
?metrics
c	variables
 ?layer_regularization_losses
?layers
dtrainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block4_conv2/kernel
 :?2block4_conv2/bias
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
hregularization_losses
?metrics
i	variables
 ?layer_regularization_losses
?layers
jtrainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block4_conv3/kernel
 :?2block4_conv3/bias
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
nregularization_losses
?metrics
o	variables
 ?layer_regularization_losses
?layers
ptrainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block4_conv4/kernel
 :?2block4_conv4/bias
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
tregularization_losses
?metrics
u	variables
 ?layer_regularization_losses
?layers
vtrainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
xregularization_losses
?metrics
y	variables
 ?layer_regularization_losses
?layers
ztrainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block5_conv1/kernel
 :?2block5_conv1/bias
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
~regularization_losses
?metrics
	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block5_conv2/kernel
 :?2block5_conv2/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block5_conv3/kernel
 :?2block5_conv3/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block5_conv4/kernel
 :?2block5_conv4/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/??2block_6_conv_1/kernel
": ?2block_6_conv_1/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/??2block_6_conv_2/kernel
": ?2block_6_conv_2/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3??2col_decoder/conv2d/kernel
&:$?2col_decoder/conv2d/bias
>:<?2#col_decoder/conv2d_transpose/kernel
/:-2!col_decoder/conv2d_transpose/bias
9:7??2table_decoder/conv2d_1/kernel
*:(?2table_decoder/conv2d_1/bias
9:7??2table_decoder/conv2d_2/kernel
*:(?2table_decoder/conv2d_2/bias
B:@?2'table_decoder/conv2d_transpose_1/kernel
3:12%table_decoder/conv2d_transpose_1/bias
?
$0
%1
*2
+3
44
55
:6
;7
D8
E9
J10
K11
P12
Q13
V14
W15
`16
a17
f18
g19
l20
m21
r22
s23
|24
}25
?26
?27
?28
?29
?30
?31"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
`
?0
?1
?2
?3
?4
?5
?6
?7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
?__inference_model_layer_call_and_return_conditional_losses_3730
?__inference_model_layer_call_and_return_conditional_losses_3991
?__inference_model_layer_call_and_return_conditional_losses_3265
?__inference_model_layer_call_and_return_conditional_losses_3389?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_1178Input_Layer"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_model_layer_call_fn_2346
$__inference_model_layer_call_fn_4090
$__inference_model_layer_call_fn_4189
$__inference_model_layer_call_fn_3141?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_block1_conv1_layer_call_and_return_conditional_losses_4200?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block1_conv1_layer_call_fn_4209?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_4220?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block1_conv2_layer_call_fn_4229?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_block1_pool_layer_call_and_return_conditional_losses_4234
E__inference_block1_pool_layer_call_and_return_conditional_losses_4239?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_block1_pool_layer_call_fn_4244
*__inference_block1_pool_layer_call_fn_4249?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block2_conv1_layer_call_and_return_conditional_losses_4260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block2_conv1_layer_call_fn_4269?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block2_conv2_layer_call_and_return_conditional_losses_4280?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block2_conv2_layer_call_fn_4289?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_block2_pool_layer_call_and_return_conditional_losses_4294
E__inference_block2_pool_layer_call_and_return_conditional_losses_4299?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_block2_pool_layer_call_fn_4304
*__inference_block2_pool_layer_call_fn_4309?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_conv1_layer_call_and_return_conditional_losses_4320?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block3_conv1_layer_call_fn_4329?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_conv2_layer_call_and_return_conditional_losses_4340?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block3_conv2_layer_call_fn_4349?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_conv3_layer_call_and_return_conditional_losses_4360?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block3_conv3_layer_call_fn_4369?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_conv4_layer_call_and_return_conditional_losses_4380?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block3_conv4_layer_call_fn_4389?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_block3_pool_layer_call_and_return_conditional_losses_4394
E__inference_block3_pool_layer_call_and_return_conditional_losses_4399?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_block3_pool_layer_call_fn_4404
*__inference_block3_pool_layer_call_fn_4409?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_conv1_layer_call_and_return_conditional_losses_4420?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block4_conv1_layer_call_fn_4429?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_conv2_layer_call_and_return_conditional_losses_4440?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block4_conv2_layer_call_fn_4449?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_conv3_layer_call_and_return_conditional_losses_4460?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block4_conv3_layer_call_fn_4469?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_conv4_layer_call_and_return_conditional_losses_4480?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block4_conv4_layer_call_fn_4489?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_block4_pool_layer_call_and_return_conditional_losses_4494
E__inference_block4_pool_layer_call_and_return_conditional_losses_4499?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_block4_pool_layer_call_fn_4504
*__inference_block4_pool_layer_call_fn_4509?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block5_conv1_layer_call_and_return_conditional_losses_4520?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block5_conv1_layer_call_fn_4529?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block5_conv2_layer_call_and_return_conditional_losses_4540?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block5_conv2_layer_call_fn_4549?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block5_conv3_layer_call_and_return_conditional_losses_4560?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block5_conv3_layer_call_fn_4569?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block5_conv4_layer_call_and_return_conditional_losses_4580?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block5_conv4_layer_call_fn_4589?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_block5_pool_layer_call_and_return_conditional_losses_4594
E__inference_block5_pool_layer_call_and_return_conditional_losses_4599?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_block5_pool_layer_call_fn_4604
*__inference_block5_pool_layer_call_fn_4609?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_4620?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_block_6_conv_1_layer_call_fn_4629?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dropout_layer_call_and_return_conditional_losses_4634
A__inference_dropout_layer_call_and_return_conditional_losses_4646?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dropout_layer_call_fn_4651
&__inference_dropout_layer_call_fn_4656?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_4667?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_block_6_conv_2_layer_call_fn_4676?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout_1_layer_call_and_return_conditional_losses_4681
C__inference_dropout_1_layer_call_and_return_conditional_losses_4693?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_1_layer_call_fn_4698
(__inference_dropout_1_layer_call_fn_4703?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_col_decoder_layer_call_and_return_conditional_losses_4756?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_col_decoder_layer_call_fn_4771?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_table_decoder_layer_call_and_return_conditional_losses_4832
G__inference_table_decoder_layer_call_and_return_conditional_losses_4900?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_table_decoder_layer_call_fn_4919
,__inference_table_decoder_layer_call_fn_4938?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_3490Input_Layer"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4950?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_up_sampling2d_layer_call_fn_4955?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_4967?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_up_sampling2d_1_layer_call_fn_4972?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_4984?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_up_sampling2d_2_layer_call_fn_4989?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_5001?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_up_sampling2d_3_layer_call_fn_5006?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_5040?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_conv2d_transpose_layer_call_fn_5049?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_5061?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_up_sampling2d_4_layer_call_fn_5066?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_5078?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_up_sampling2d_5_layer_call_fn_5083?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_5095?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_up_sampling2d_6_layer_call_fn_5100?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_5112?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_up_sampling2d_7_layer_call_fn_5117?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_5151?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_conv2d_transpose_1_layer_call_fn_5160?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_1178?B$%*+45:;DEJKPQVW`afglmrs|}????????????????????>?;
4?1
/?,
Input_Layer???????????
? "???
>
col_decoder/?,
col_decoder???????????
B
table_decoder1?.
table_decoder????????????
F__inference_block1_conv1_layer_call_and_return_conditional_losses_4200p$%9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????@
? ?
+__inference_block1_conv1_layer_call_fn_4209c$%9?6
/?,
*?'
inputs???????????
? ""????????????@?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_4220p*+9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
+__inference_block1_conv2_layer_call_fn_4229c*+9?6
/?,
*?'
inputs???????????@
? ""????????????@?
E__inference_block1_pool_layer_call_and_return_conditional_losses_4234?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
E__inference_block1_pool_layer_call_and_return_conditional_losses_4239l9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
*__inference_block1_pool_layer_call_fn_4244?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
*__inference_block1_pool_layer_call_fn_4249_9?6
/?,
*?'
inputs???????????@
? ""????????????@?
F__inference_block2_conv1_layer_call_and_return_conditional_losses_4260q459?6
/?,
*?'
inputs???????????@
? "0?-
&?#
0????????????
? ?
+__inference_block2_conv1_layer_call_fn_4269d459?6
/?,
*?'
inputs???????????@
? "#? ?????????????
F__inference_block2_conv2_layer_call_and_return_conditional_losses_4280r:;:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
+__inference_block2_conv2_layer_call_fn_4289e:;:?7
0?-
+?(
inputs????????????
? "#? ?????????????
E__inference_block2_pool_layer_call_and_return_conditional_losses_4294?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
E__inference_block2_pool_layer_call_and_return_conditional_losses_4299n:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
*__inference_block2_pool_layer_call_fn_4304?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
*__inference_block2_pool_layer_call_fn_4309a:?7
0?-
+?(
inputs????????????
? "#? ?????????????
F__inference_block3_conv1_layer_call_and_return_conditional_losses_4320rDE:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
+__inference_block3_conv1_layer_call_fn_4329eDE:?7
0?-
+?(
inputs????????????
? "#? ?????????????
F__inference_block3_conv2_layer_call_and_return_conditional_losses_4340rJK:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
+__inference_block3_conv2_layer_call_fn_4349eJK:?7
0?-
+?(
inputs????????????
? "#? ?????????????
F__inference_block3_conv3_layer_call_and_return_conditional_losses_4360rPQ:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
+__inference_block3_conv3_layer_call_fn_4369ePQ:?7
0?-
+?(
inputs????????????
? "#? ?????????????
F__inference_block3_conv4_layer_call_and_return_conditional_losses_4380rVW:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
+__inference_block3_conv4_layer_call_fn_4389eVW:?7
0?-
+?(
inputs????????????
? "#? ?????????????
E__inference_block3_pool_layer_call_and_return_conditional_losses_4394?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
E__inference_block3_pool_layer_call_and_return_conditional_losses_4399l:?7
0?-
+?(
inputs????????????
? ".?+
$?!
0?????????dd?
? ?
*__inference_block3_pool_layer_call_fn_4404?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
*__inference_block3_pool_layer_call_fn_4409_:?7
0?-
+?(
inputs????????????
? "!??????????dd??
F__inference_block4_conv1_layer_call_and_return_conditional_losses_4420n`a8?5
.?+
)?&
inputs?????????dd?
? ".?+
$?!
0?????????dd?
? ?
+__inference_block4_conv1_layer_call_fn_4429a`a8?5
.?+
)?&
inputs?????????dd?
? "!??????????dd??
F__inference_block4_conv2_layer_call_and_return_conditional_losses_4440nfg8?5
.?+
)?&
inputs?????????dd?
? ".?+
$?!
0?????????dd?
? ?
+__inference_block4_conv2_layer_call_fn_4449afg8?5
.?+
)?&
inputs?????????dd?
? "!??????????dd??
F__inference_block4_conv3_layer_call_and_return_conditional_losses_4460nlm8?5
.?+
)?&
inputs?????????dd?
? ".?+
$?!
0?????????dd?
? ?
+__inference_block4_conv3_layer_call_fn_4469alm8?5
.?+
)?&
inputs?????????dd?
? "!??????????dd??
F__inference_block4_conv4_layer_call_and_return_conditional_losses_4480nrs8?5
.?+
)?&
inputs?????????dd?
? ".?+
$?!
0?????????dd?
? ?
+__inference_block4_conv4_layer_call_fn_4489ars8?5
.?+
)?&
inputs?????????dd?
? "!??????????dd??
E__inference_block4_pool_layer_call_and_return_conditional_losses_4494?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
E__inference_block4_pool_layer_call_and_return_conditional_losses_4499j8?5
.?+
)?&
inputs?????????dd?
? ".?+
$?!
0?????????22?
? ?
*__inference_block4_pool_layer_call_fn_4504?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
*__inference_block4_pool_layer_call_fn_4509]8?5
.?+
)?&
inputs?????????dd?
? "!??????????22??
F__inference_block5_conv1_layer_call_and_return_conditional_losses_4520n|}8?5
.?+
)?&
inputs?????????22?
? ".?+
$?!
0?????????22?
? ?
+__inference_block5_conv1_layer_call_fn_4529a|}8?5
.?+
)?&
inputs?????????22?
? "!??????????22??
F__inference_block5_conv2_layer_call_and_return_conditional_losses_4540p??8?5
.?+
)?&
inputs?????????22?
? ".?+
$?!
0?????????22?
? ?
+__inference_block5_conv2_layer_call_fn_4549c??8?5
.?+
)?&
inputs?????????22?
? "!??????????22??
F__inference_block5_conv3_layer_call_and_return_conditional_losses_4560p??8?5
.?+
)?&
inputs?????????22?
? ".?+
$?!
0?????????22?
? ?
+__inference_block5_conv3_layer_call_fn_4569c??8?5
.?+
)?&
inputs?????????22?
? "!??????????22??
F__inference_block5_conv4_layer_call_and_return_conditional_losses_4580p??8?5
.?+
)?&
inputs?????????22?
? ".?+
$?!
0?????????22?
? ?
+__inference_block5_conv4_layer_call_fn_4589c??8?5
.?+
)?&
inputs?????????22?
? "!??????????22??
E__inference_block5_pool_layer_call_and_return_conditional_losses_4594?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
E__inference_block5_pool_layer_call_and_return_conditional_losses_4599j8?5
.?+
)?&
inputs?????????22?
? ".?+
$?!
0??????????
? ?
*__inference_block5_pool_layer_call_fn_4604?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
*__inference_block5_pool_layer_call_fn_4609]8?5
.?+
)?&
inputs?????????22?
? "!????????????
H__inference_block_6_conv_1_layer_call_and_return_conditional_losses_4620p??8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_block_6_conv_1_layer_call_fn_4629c??8?5
.?+
)?&
inputs??????????
? "!????????????
H__inference_block_6_conv_2_layer_call_and_return_conditional_losses_4667p??8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_block_6_conv_2_layer_call_fn_4676c??8?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_col_decoder_layer_call_and_return_conditional_losses_4756????????
???
???
*?'
input/0??????????
*?'
input/1?????????22?
*?'
input/2?????????dd?
? "/?,
%?"
0???????????
? ?
*__inference_col_decoder_layer_call_fn_4771????????
???
???
*?'
input/0??????????
*?'
input/1?????????22?
*?'
input/2?????????dd?
? ""?????????????
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_5151???J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
1__inference_conv2d_transpose_1_layer_call_fn_5160???J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+????????????????????????????
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_5040???J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
/__inference_conv2d_transpose_layer_call_fn_5049???J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+????????????????????????????
C__inference_dropout_1_layer_call_and_return_conditional_losses_4681n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_4693n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
(__inference_dropout_1_layer_call_fn_4698a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
(__inference_dropout_1_layer_call_fn_4703a<?9
2?/
)?&
inputs??????????
p
? "!????????????
A__inference_dropout_layer_call_and_return_conditional_losses_4634n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_4646n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
&__inference_dropout_layer_call_fn_4651a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
&__inference_dropout_layer_call_fn_4656a<?9
2?/
)?&
inputs??????????
p
? "!????????????
?__inference_model_layer_call_and_return_conditional_losses_3265?B$%*+45:;DEJKPQVW`afglmrs|}????????????????????F?C
<?9
/?,
Input_Layer???????????
p 

 
? "_?\
U?R
'?$
0/0???????????
'?$
0/1???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_3389?B$%*+45:;DEJKPQVW`afglmrs|}????????????????????F?C
<?9
/?,
Input_Layer???????????
p

 
? "_?\
U?R
'?$
0/0???????????
'?$
0/1???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_3730?B$%*+45:;DEJKPQVW`afglmrs|}????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "_?\
U?R
'?$
0/0???????????
'?$
0/1???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_3991?B$%*+45:;DEJKPQVW`afglmrs|}????????????????????A?>
7?4
*?'
inputs???????????
p

 
? "_?\
U?R
'?$
0/0???????????
'?$
0/1???????????
? ?
$__inference_model_layer_call_fn_2346?B$%*+45:;DEJKPQVW`afglmrs|}????????????????????F?C
<?9
/?,
Input_Layer???????????
p 

 
? "Q?N
%?"
0???????????
%?"
1????????????
$__inference_model_layer_call_fn_3141?B$%*+45:;DEJKPQVW`afglmrs|}????????????????????F?C
<?9
/?,
Input_Layer???????????
p

 
? "Q?N
%?"
0???????????
%?"
1????????????
$__inference_model_layer_call_fn_4090?B$%*+45:;DEJKPQVW`afglmrs|}????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "Q?N
%?"
0???????????
%?"
1????????????
$__inference_model_layer_call_fn_4189?B$%*+45:;DEJKPQVW`afglmrs|}????????????????????A?>
7?4
*?'
inputs???????????
p

 
? "Q?N
%?"
0???????????
%?"
1????????????
"__inference_signature_wrapper_3490?B$%*+45:;DEJKPQVW`afglmrs|}????????????????????M?J
? 
C?@
>
Input_Layer/?,
Input_Layer???????????"???
>
col_decoder/?,
col_decoder???????????
B
table_decoder1?.
table_decoder????????????
G__inference_table_decoder_layer_call_and_return_conditional_losses_4832??????????
???
???
*?'
input/0??????????
*?'
input/1?????????22?
*?'
input/2?????????dd?
p 
? "/?,
%?"
0???????????
? ?
G__inference_table_decoder_layer_call_and_return_conditional_losses_4900??????????
???
???
*?'
input/0??????????
*?'
input/1?????????22?
*?'
input/2?????????dd?
p
? "/?,
%?"
0???????????
? ?
,__inference_table_decoder_layer_call_fn_4919??????????
???
???
*?'
input/0??????????
*?'
input/1?????????22?
*?'
input/2?????????dd?
p 
? ""?????????????
,__inference_table_decoder_layer_call_fn_4938??????????
???
???
*?'
input/0??????????
*?'
input/1?????????22?
*?'
input/2?????????dd?
p
? ""?????????????
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_4967?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_up_sampling2d_1_layer_call_fn_4972?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_4984?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_up_sampling2d_2_layer_call_fn_4989?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_5001?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_up_sampling2d_3_layer_call_fn_5006?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_5061?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_up_sampling2d_4_layer_call_fn_5066?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_5078?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_up_sampling2d_5_layer_call_fn_5083?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_5095?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_up_sampling2d_6_layer_call_fn_5100?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_5112?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_up_sampling2d_7_layer_call_fn_5117?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4950?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_up_sampling2d_layer_call_fn_4955?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????