ś
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8Ԃ
�
numeric_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_namenumeric_embedding/embeddings
�
0numeric_embedding/embeddings/Read/ReadVariableOpReadVariableOpnumeric_embedding/embeddings*
_output_shapes

:*
dtype0
�
shift_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameshift_embedding/embeddings
�
.shift_embedding/embeddings/Read/ReadVariableOpReadVariableOpshift_embedding/embeddings*
_output_shapes

:*
dtype0
�
quality_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_namequality_embedding/embeddings
�
0quality_embedding/embeddings/Read/ReadVariableOpReadVariableOpquality_embedding/embeddings*
_output_shapes

:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
�
numeric_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_namenumeric_output/kernel
�
)numeric_output/kernel/Read/ReadVariableOpReadVariableOpnumeric_output/kernel*
_output_shapes
:	�*
dtype0
~
numeric_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namenumeric_output/bias
w
'numeric_output/bias/Read/ReadVariableOpReadVariableOpnumeric_output/bias*
_output_shapes
:*
dtype0
�
shift_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameshift_output/kernel
|
'shift_output/kernel/Read/ReadVariableOpReadVariableOpshift_output/kernel*
_output_shapes
:	�*
dtype0
z
shift_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameshift_output/bias
s
%shift_output/bias/Read/ReadVariableOpReadVariableOpshift_output/bias*
_output_shapes
:*
dtype0
�
quality_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_namequality_output/kernel
�
)quality_output/kernel/Read/ReadVariableOpReadVariableOpquality_output/kernel*
_output_shapes
:	�*
dtype0
~
quality_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namequality_output/bias
w
'quality_output/bias/Read/ReadVariableOpReadVariableOpquality_output/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
�
#Adam/numeric_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/numeric_embedding/embeddings/m
�
7Adam/numeric_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp#Adam/numeric_embedding/embeddings/m*
_output_shapes

:*
dtype0
�
!Adam/shift_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/shift_embedding/embeddings/m
�
5Adam/shift_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp!Adam/shift_embedding/embeddings/m*
_output_shapes

:*
dtype0
�
#Adam/quality_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/quality_embedding/embeddings/m
�
7Adam/quality_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp#Adam/quality_embedding/embeddings/m*
_output_shapes

:*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	�*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/numeric_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_nameAdam/numeric_output/kernel/m
�
0Adam/numeric_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/numeric_output/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/numeric_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/numeric_output/bias/m
�
.Adam/numeric_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/numeric_output/bias/m*
_output_shapes
:*
dtype0
�
Adam/shift_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameAdam/shift_output/kernel/m
�
.Adam/shift_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/shift_output/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/shift_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/shift_output/bias/m
�
,Adam/shift_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/shift_output/bias/m*
_output_shapes
:*
dtype0
�
Adam/quality_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_nameAdam/quality_output/kernel/m
�
0Adam/quality_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quality_output/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/quality_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/quality_output/bias/m
�
.Adam/quality_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/quality_output/bias/m*
_output_shapes
:*
dtype0
�
#Adam/numeric_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/numeric_embedding/embeddings/v
�
7Adam/numeric_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp#Adam/numeric_embedding/embeddings/v*
_output_shapes

:*
dtype0
�
!Adam/shift_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/shift_embedding/embeddings/v
�
5Adam/shift_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp!Adam/shift_embedding/embeddings/v*
_output_shapes

:*
dtype0
�
#Adam/quality_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/quality_embedding/embeddings/v
�
7Adam/quality_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp#Adam/quality_embedding/embeddings/v*
_output_shapes

:*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	�*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/numeric_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_nameAdam/numeric_output/kernel/v
�
0Adam/numeric_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/numeric_output/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/numeric_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/numeric_output/bias/v
�
.Adam/numeric_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/numeric_output/bias/v*
_output_shapes
:*
dtype0
�
Adam/shift_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameAdam/shift_output/kernel/v
�
.Adam/shift_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/shift_output/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/shift_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/shift_output/bias/v
�
,Adam/shift_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/shift_output/bias/v*
_output_shapes
:*
dtype0
�
Adam/quality_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_nameAdam/quality_output/kernel/v
�
0Adam/quality_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quality_output/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/quality_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/quality_output/bias/v
�
.Adam/quality_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/quality_output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�U
value�UB�U B�U
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
 
 
b

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
b

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
b

embeddings
 trainable_variables
!	variables
"regularization_losses
#	keras_api
R
$trainable_variables
%	variables
&regularization_losses
'	keras_api
R
(trainable_variables
)	variables
*regularization_losses
+	keras_api
R
,trainable_variables
-	variables
.regularization_losses
/	keras_api
R
0trainable_variables
1	variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
h

@kernel
Abias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
h

Fkernel
Gbias
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
�
Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_ratem�m�m�4m�5m�:m�;m�@m�Am�Fm�Gm�v�v�v�4v�5v�:v�;v�@v�Av�Fv�Gv�
N
0
1
2
43
54
:5
;6
@7
A8
F9
G10
N
0
1
2
43
54
:5
;6
@7
A8
F9
G10
 
�
Qmetrics
trainable_variables
Rlayer_regularization_losses
	variables

Slayers
regularization_losses
Tlayer_metrics
Unon_trainable_variables
 
lj
VARIABLE_VALUEnumeric_embedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�
Vmetrics
trainable_variables
Wlayer_regularization_losses
	variables

Xlayers
regularization_losses
Ylayer_metrics
Znon_trainable_variables
jh
VARIABLE_VALUEshift_embedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�
[metrics
trainable_variables
\layer_regularization_losses
	variables

]layers
regularization_losses
^layer_metrics
_non_trainable_variables
lj
VARIABLE_VALUEquality_embedding/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�
`metrics
 trainable_variables
alayer_regularization_losses
!	variables

blayers
"regularization_losses
clayer_metrics
dnon_trainable_variables
 
 
 
�
emetrics
$trainable_variables
flayer_regularization_losses
%	variables

glayers
&regularization_losses
hlayer_metrics
inon_trainable_variables
 
 
 
�
jmetrics
(trainable_variables
klayer_regularization_losses
)	variables

llayers
*regularization_losses
mlayer_metrics
nnon_trainable_variables
 
 
 
�
ometrics
,trainable_variables
player_regularization_losses
-	variables

qlayers
.regularization_losses
rlayer_metrics
snon_trainable_variables
 
 
 
�
tmetrics
0trainable_variables
ulayer_regularization_losses
1	variables

vlayers
2regularization_losses
wlayer_metrics
xnon_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
�
ymetrics
6trainable_variables
zlayer_regularization_losses
7	variables

{layers
8regularization_losses
|layer_metrics
}non_trainable_variables
a_
VARIABLE_VALUEnumeric_output/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEnumeric_output/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
�
~metrics
<trainable_variables
layer_regularization_losses
=	variables
�layers
>regularization_losses
�layer_metrics
�non_trainable_variables
_]
VARIABLE_VALUEshift_output/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEshift_output/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1

@0
A1
 
�
�metrics
Btrainable_variables
 �layer_regularization_losses
C	variables
�layers
Dregularization_losses
�layer_metrics
�non_trainable_variables
a_
VARIABLE_VALUEquality_output/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEquality_output/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

F0
G1
 
�
�metrics
Htrainable_variables
 �layer_regularization_losses
I	variables
�layers
Jregularization_losses
�layer_metrics
�non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
8
�0
�1
�2
�3
�4
�5
�6
 
f
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
8

�total

�count
�	variables
�	keras_api
8

�total

�count
�	variables
�	keras_api
8

�total

�count
�	variables
�	keras_api
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_54keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_54keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_64keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
��
VARIABLE_VALUE#Adam/numeric_embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/shift_embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/quality_embedding/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/numeric_output/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/numeric_output/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/shift_output/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/shift_output/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/quality_output/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/quality_output/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/numeric_embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/shift_embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/quality_embedding/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/numeric_output/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/numeric_output/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/shift_output/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/shift_output/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/quality_output/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/quality_output/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_numeric_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
serving_default_quality_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
~
serving_default_shift_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_numeric_inputserving_default_quality_inputserving_default_shift_inputquality_embedding/embeddingsshift_embedding/embeddingsnumeric_embedding/embeddingsdense/kernel
dense/biasquality_output/kernelquality_output/biasshift_output/kernelshift_output/biasnumeric_output/kernelnumeric_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1774236
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0numeric_embedding/embeddings/Read/ReadVariableOp.shift_embedding/embeddings/Read/ReadVariableOp0quality_embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp)numeric_output/kernel/Read/ReadVariableOp'numeric_output/bias/Read/ReadVariableOp'shift_output/kernel/Read/ReadVariableOp%shift_output/bias/Read/ReadVariableOp)quality_output/kernel/Read/ReadVariableOp'quality_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOp7Adam/numeric_embedding/embeddings/m/Read/ReadVariableOp5Adam/shift_embedding/embeddings/m/Read/ReadVariableOp7Adam/quality_embedding/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp0Adam/numeric_output/kernel/m/Read/ReadVariableOp.Adam/numeric_output/bias/m/Read/ReadVariableOp.Adam/shift_output/kernel/m/Read/ReadVariableOp,Adam/shift_output/bias/m/Read/ReadVariableOp0Adam/quality_output/kernel/m/Read/ReadVariableOp.Adam/quality_output/bias/m/Read/ReadVariableOp7Adam/numeric_embedding/embeddings/v/Read/ReadVariableOp5Adam/shift_embedding/embeddings/v/Read/ReadVariableOp7Adam/quality_embedding/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp0Adam/numeric_output/kernel/v/Read/ReadVariableOp.Adam/numeric_output/bias/v/Read/ReadVariableOp.Adam/shift_output/kernel/v/Read/ReadVariableOp,Adam/shift_output/bias/v/Read/ReadVariableOp0Adam/quality_output/kernel/v/Read/ReadVariableOp.Adam/quality_output/bias/v/Read/ReadVariableOpConst*A
Tin:
826	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_1774788
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenumeric_embedding/embeddingsshift_embedding/embeddingsquality_embedding/embeddingsdense/kernel
dense/biasnumeric_output/kernelnumeric_output/biasshift_output/kernelshift_output/biasquality_output/kernelquality_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4total_5count_5total_6count_6#Adam/numeric_embedding/embeddings/m!Adam/shift_embedding/embeddings/m#Adam/quality_embedding/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/numeric_output/kernel/mAdam/numeric_output/bias/mAdam/shift_output/kernel/mAdam/shift_output/bias/mAdam/quality_output/kernel/mAdam/quality_output/bias/m#Adam/numeric_embedding/embeddings/v!Adam/shift_embedding/embeddings/v#Adam/quality_embedding/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/numeric_output/kernel/vAdam/numeric_output/bias/vAdam/shift_output/kernel/vAdam/shift_output/bias/vAdam/quality_output/kernel/vAdam/quality_output/bias/v*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_1774954��	
�
w
1__inference_shift_embedding_layer_call_fn_1774460

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_shift_embedding_layer_call_and_return_conditional_losses_17737982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_harmony_model_layer_call_fn_1774393
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_harmony_model_layer_call_and_return_conditional_losses_17740902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�<
�
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774090

inputs
inputs_1
inputs_2
quality_embedding_1774054
shift_embedding_1774057
numeric_embedding_1774060
dense_1774067
dense_1774069
quality_output_1774072
quality_output_1774074
shift_output_1774077
shift_output_1774079
numeric_output_1774082
numeric_output_1774084
identity

identity_1

identity_2��dense/StatefulPartitionedCall�)numeric_embedding/StatefulPartitionedCall�&numeric_output/StatefulPartitionedCall�)quality_embedding/StatefulPartitionedCall�&quality_output/StatefulPartitionedCall�'shift_embedding/StatefulPartitionedCall�$shift_output/StatefulPartitionedCall�
)quality_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_2quality_embedding_1774054*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_quality_embedding_layer_call_and_return_conditional_losses_17737762+
)quality_embedding/StatefulPartitionedCall�
'shift_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1shift_embedding_1774057*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_shift_embedding_layer_call_and_return_conditional_losses_17737982)
'shift_embedding/StatefulPartitionedCall�
)numeric_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsnumeric_embedding_1774060*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_numeric_embedding_layer_call_and_return_conditional_losses_17738202+
)numeric_embedding/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall2numeric_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17738382
flatten/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall0shift_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_17738522
flatten_1/PartitionedCall�
flatten_2/PartitionedCallPartitionedCall2quality_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_17738662
flatten_2/PartitionedCall�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_17738822
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1774067dense_1774069*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_17739032
dense/StatefulPartitionedCall�
&quality_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0quality_output_1774072quality_output_1774074*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quality_output_layer_call_and_return_conditional_losses_17739302(
&quality_output/StatefulPartitionedCall�
$shift_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0shift_output_1774077shift_output_1774079*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_shift_output_layer_call_and_return_conditional_losses_17739572&
$shift_output/StatefulPartitionedCall�
&numeric_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0numeric_output_1774082numeric_output_1774084*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_numeric_output_layer_call_and_return_conditional_losses_17739842(
&numeric_output/StatefulPartitionedCall�
IdentityIdentity/numeric_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity-shift_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity/quality_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2V
)numeric_embedding/StatefulPartitionedCall)numeric_embedding/StatefulPartitionedCall2P
&numeric_output/StatefulPartitionedCall&numeric_output/StatefulPartitionedCall2V
)quality_embedding/StatefulPartitionedCall)quality_embedding/StatefulPartitionedCall2P
&quality_output/StatefulPartitionedCall&quality_output/StatefulPartitionedCall2R
'shift_embedding/StatefulPartitionedCall'shift_embedding/StatefulPartitionedCall2L
$shift_output/StatefulPartitionedCall$shift_output/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�Z
�
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774360
inputs_0
inputs_1
inputs_2.
*quality_embedding_embedding_lookup_1774304,
(shift_embedding_embedding_lookup_1774310.
*numeric_embedding_embedding_lookup_1774316(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource1
-quality_output_matmul_readvariableop_resource2
.quality_output_biasadd_readvariableop_resource/
+shift_output_matmul_readvariableop_resource0
,shift_output_biasadd_readvariableop_resource1
-numeric_output_matmul_readvariableop_resource2
.numeric_output_biasadd_readvariableop_resource
identity

identity_1

identity_2��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�"numeric_embedding/embedding_lookup�%numeric_output/BiasAdd/ReadVariableOp�$numeric_output/MatMul/ReadVariableOp�"quality_embedding/embedding_lookup�%quality_output/BiasAdd/ReadVariableOp�$quality_output/MatMul/ReadVariableOp� shift_embedding/embedding_lookup�#shift_output/BiasAdd/ReadVariableOp�"shift_output/MatMul/ReadVariableOp�
quality_embedding/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:���������2
quality_embedding/Cast�
"quality_embedding/embedding_lookupResourceGather*quality_embedding_embedding_lookup_1774304quality_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*=
_class3
1/loc:@quality_embedding/embedding_lookup/1774304*+
_output_shapes
:���������*
dtype02$
"quality_embedding/embedding_lookup�
+quality_embedding/embedding_lookup/IdentityIdentity+quality_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@quality_embedding/embedding_lookup/1774304*+
_output_shapes
:���������2-
+quality_embedding/embedding_lookup/Identity�
-quality_embedding/embedding_lookup/Identity_1Identity4quality_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2/
-quality_embedding/embedding_lookup/Identity_1
shift_embedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:���������2
shift_embedding/Cast�
 shift_embedding/embedding_lookupResourceGather(shift_embedding_embedding_lookup_1774310shift_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@shift_embedding/embedding_lookup/1774310*+
_output_shapes
:���������*
dtype02"
 shift_embedding/embedding_lookup�
)shift_embedding/embedding_lookup/IdentityIdentity)shift_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@shift_embedding/embedding_lookup/1774310*+
_output_shapes
:���������2+
)shift_embedding/embedding_lookup/Identity�
+shift_embedding/embedding_lookup/Identity_1Identity2shift_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2-
+shift_embedding/embedding_lookup/Identity_1�
numeric_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������2
numeric_embedding/Cast�
"numeric_embedding/embedding_lookupResourceGather*numeric_embedding_embedding_lookup_1774316numeric_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*=
_class3
1/loc:@numeric_embedding/embedding_lookup/1774316*+
_output_shapes
:���������*
dtype02$
"numeric_embedding/embedding_lookup�
+numeric_embedding/embedding_lookup/IdentityIdentity+numeric_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@numeric_embedding/embedding_lookup/1774316*+
_output_shapes
:���������2-
+numeric_embedding/embedding_lookup/Identity�
-numeric_embedding/embedding_lookup/Identity_1Identity4numeric_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2/
-numeric_embedding/embedding_lookup/Identity_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten/Const�
flatten/ReshapeReshape6numeric_embedding/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:���������2
flatten/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_1/Const�
flatten_1/ReshapeReshape4shift_embedding/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:���������2
flatten_1/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_2/Const�
flatten_2/ReshapeReshape6quality_embedding/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:���������2
flatten_2/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate/concat�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
$quality_output/MatMul/ReadVariableOpReadVariableOp-quality_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02&
$quality_output/MatMul/ReadVariableOp�
quality_output/MatMulMatMuldense/Relu:activations:0,quality_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
quality_output/MatMul�
%quality_output/BiasAdd/ReadVariableOpReadVariableOp.quality_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quality_output/BiasAdd/ReadVariableOp�
quality_output/BiasAddBiasAddquality_output/MatMul:product:0-quality_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
quality_output/BiasAdd�
quality_output/SoftmaxSoftmaxquality_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
quality_output/Softmax�
"shift_output/MatMul/ReadVariableOpReadVariableOp+shift_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"shift_output/MatMul/ReadVariableOp�
shift_output/MatMulMatMuldense/Relu:activations:0*shift_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
shift_output/MatMul�
#shift_output/BiasAdd/ReadVariableOpReadVariableOp,shift_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#shift_output/BiasAdd/ReadVariableOp�
shift_output/BiasAddBiasAddshift_output/MatMul:product:0+shift_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
shift_output/BiasAdd�
shift_output/SoftmaxSoftmaxshift_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
shift_output/Softmax�
$numeric_output/MatMul/ReadVariableOpReadVariableOp-numeric_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02&
$numeric_output/MatMul/ReadVariableOp�
numeric_output/MatMulMatMuldense/Relu:activations:0,numeric_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
numeric_output/MatMul�
%numeric_output/BiasAdd/ReadVariableOpReadVariableOp.numeric_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%numeric_output/BiasAdd/ReadVariableOp�
numeric_output/BiasAddBiasAddnumeric_output/MatMul:product:0-numeric_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
numeric_output/BiasAdd�
numeric_output/SoftmaxSoftmaxnumeric_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
numeric_output/Softmax�
IdentityIdentity numeric_output/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^numeric_embedding/embedding_lookup&^numeric_output/BiasAdd/ReadVariableOp%^numeric_output/MatMul/ReadVariableOp#^quality_embedding/embedding_lookup&^quality_output/BiasAdd/ReadVariableOp%^quality_output/MatMul/ReadVariableOp!^shift_embedding/embedding_lookup$^shift_output/BiasAdd/ReadVariableOp#^shift_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identityshift_output/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^numeric_embedding/embedding_lookup&^numeric_output/BiasAdd/ReadVariableOp%^numeric_output/MatMul/ReadVariableOp#^quality_embedding/embedding_lookup&^quality_output/BiasAdd/ReadVariableOp%^quality_output/MatMul/ReadVariableOp!^shift_embedding/embedding_lookup$^shift_output/BiasAdd/ReadVariableOp#^shift_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity quality_output/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^numeric_embedding/embedding_lookup&^numeric_output/BiasAdd/ReadVariableOp%^numeric_output/MatMul/ReadVariableOp#^quality_embedding/embedding_lookup&^quality_output/BiasAdd/ReadVariableOp%^quality_output/MatMul/ReadVariableOp!^shift_embedding/embedding_lookup$^shift_output/BiasAdd/ReadVariableOp#^shift_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2H
"numeric_embedding/embedding_lookup"numeric_embedding/embedding_lookup2N
%numeric_output/BiasAdd/ReadVariableOp%numeric_output/BiasAdd/ReadVariableOp2L
$numeric_output/MatMul/ReadVariableOp$numeric_output/MatMul/ReadVariableOp2H
"quality_embedding/embedding_lookup"quality_embedding/embedding_lookup2N
%quality_output/BiasAdd/ReadVariableOp%quality_output/BiasAdd/ReadVariableOp2L
$quality_output/MatMul/ReadVariableOp$quality_output/MatMul/ReadVariableOp2D
 shift_embedding/embedding_lookup shift_embedding/embedding_lookup2J
#shift_output/BiasAdd/ReadVariableOp#shift_output/BiasAdd/ReadVariableOp2H
"shift_output/MatMul/ReadVariableOp"shift_output/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1773852

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_quality_output_layer_call_fn_1774605

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quality_output_layer_call_and_return_conditional_losses_17739302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�Z
�
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774298
inputs_0
inputs_1
inputs_2.
*quality_embedding_embedding_lookup_1774242,
(shift_embedding_embedding_lookup_1774248.
*numeric_embedding_embedding_lookup_1774254(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource1
-quality_output_matmul_readvariableop_resource2
.quality_output_biasadd_readvariableop_resource/
+shift_output_matmul_readvariableop_resource0
,shift_output_biasadd_readvariableop_resource1
-numeric_output_matmul_readvariableop_resource2
.numeric_output_biasadd_readvariableop_resource
identity

identity_1

identity_2��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�"numeric_embedding/embedding_lookup�%numeric_output/BiasAdd/ReadVariableOp�$numeric_output/MatMul/ReadVariableOp�"quality_embedding/embedding_lookup�%quality_output/BiasAdd/ReadVariableOp�$quality_output/MatMul/ReadVariableOp� shift_embedding/embedding_lookup�#shift_output/BiasAdd/ReadVariableOp�"shift_output/MatMul/ReadVariableOp�
quality_embedding/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:���������2
quality_embedding/Cast�
"quality_embedding/embedding_lookupResourceGather*quality_embedding_embedding_lookup_1774242quality_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*=
_class3
1/loc:@quality_embedding/embedding_lookup/1774242*+
_output_shapes
:���������*
dtype02$
"quality_embedding/embedding_lookup�
+quality_embedding/embedding_lookup/IdentityIdentity+quality_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@quality_embedding/embedding_lookup/1774242*+
_output_shapes
:���������2-
+quality_embedding/embedding_lookup/Identity�
-quality_embedding/embedding_lookup/Identity_1Identity4quality_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2/
-quality_embedding/embedding_lookup/Identity_1
shift_embedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:���������2
shift_embedding/Cast�
 shift_embedding/embedding_lookupResourceGather(shift_embedding_embedding_lookup_1774248shift_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@shift_embedding/embedding_lookup/1774248*+
_output_shapes
:���������*
dtype02"
 shift_embedding/embedding_lookup�
)shift_embedding/embedding_lookup/IdentityIdentity)shift_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@shift_embedding/embedding_lookup/1774248*+
_output_shapes
:���������2+
)shift_embedding/embedding_lookup/Identity�
+shift_embedding/embedding_lookup/Identity_1Identity2shift_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2-
+shift_embedding/embedding_lookup/Identity_1�
numeric_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������2
numeric_embedding/Cast�
"numeric_embedding/embedding_lookupResourceGather*numeric_embedding_embedding_lookup_1774254numeric_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*=
_class3
1/loc:@numeric_embedding/embedding_lookup/1774254*+
_output_shapes
:���������*
dtype02$
"numeric_embedding/embedding_lookup�
+numeric_embedding/embedding_lookup/IdentityIdentity+numeric_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@numeric_embedding/embedding_lookup/1774254*+
_output_shapes
:���������2-
+numeric_embedding/embedding_lookup/Identity�
-numeric_embedding/embedding_lookup/Identity_1Identity4numeric_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2/
-numeric_embedding/embedding_lookup/Identity_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten/Const�
flatten/ReshapeReshape6numeric_embedding/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:���������2
flatten/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_1/Const�
flatten_1/ReshapeReshape4shift_embedding/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:���������2
flatten_1/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_2/Const�
flatten_2/ReshapeReshape6quality_embedding/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:���������2
flatten_2/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate/concat�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
$quality_output/MatMul/ReadVariableOpReadVariableOp-quality_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02&
$quality_output/MatMul/ReadVariableOp�
quality_output/MatMulMatMuldense/Relu:activations:0,quality_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
quality_output/MatMul�
%quality_output/BiasAdd/ReadVariableOpReadVariableOp.quality_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quality_output/BiasAdd/ReadVariableOp�
quality_output/BiasAddBiasAddquality_output/MatMul:product:0-quality_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
quality_output/BiasAdd�
quality_output/SoftmaxSoftmaxquality_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
quality_output/Softmax�
"shift_output/MatMul/ReadVariableOpReadVariableOp+shift_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"shift_output/MatMul/ReadVariableOp�
shift_output/MatMulMatMuldense/Relu:activations:0*shift_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
shift_output/MatMul�
#shift_output/BiasAdd/ReadVariableOpReadVariableOp,shift_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#shift_output/BiasAdd/ReadVariableOp�
shift_output/BiasAddBiasAddshift_output/MatMul:product:0+shift_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
shift_output/BiasAdd�
shift_output/SoftmaxSoftmaxshift_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
shift_output/Softmax�
$numeric_output/MatMul/ReadVariableOpReadVariableOp-numeric_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02&
$numeric_output/MatMul/ReadVariableOp�
numeric_output/MatMulMatMuldense/Relu:activations:0,numeric_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
numeric_output/MatMul�
%numeric_output/BiasAdd/ReadVariableOpReadVariableOp.numeric_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%numeric_output/BiasAdd/ReadVariableOp�
numeric_output/BiasAddBiasAddnumeric_output/MatMul:product:0-numeric_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
numeric_output/BiasAdd�
numeric_output/SoftmaxSoftmaxnumeric_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
numeric_output/Softmax�
IdentityIdentity numeric_output/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^numeric_embedding/embedding_lookup&^numeric_output/BiasAdd/ReadVariableOp%^numeric_output/MatMul/ReadVariableOp#^quality_embedding/embedding_lookup&^quality_output/BiasAdd/ReadVariableOp%^quality_output/MatMul/ReadVariableOp!^shift_embedding/embedding_lookup$^shift_output/BiasAdd/ReadVariableOp#^shift_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identityshift_output/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^numeric_embedding/embedding_lookup&^numeric_output/BiasAdd/ReadVariableOp%^numeric_output/MatMul/ReadVariableOp#^quality_embedding/embedding_lookup&^quality_output/BiasAdd/ReadVariableOp%^quality_output/MatMul/ReadVariableOp!^shift_embedding/embedding_lookup$^shift_output/BiasAdd/ReadVariableOp#^shift_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity quality_output/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^numeric_embedding/embedding_lookup&^numeric_output/BiasAdd/ReadVariableOp%^numeric_output/MatMul/ReadVariableOp#^quality_embedding/embedding_lookup&^quality_output/BiasAdd/ReadVariableOp%^quality_output/MatMul/ReadVariableOp!^shift_embedding/embedding_lookup$^shift_output/BiasAdd/ReadVariableOp#^shift_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2H
"numeric_embedding/embedding_lookup"numeric_embedding/embedding_lookup2N
%numeric_output/BiasAdd/ReadVariableOp%numeric_output/BiasAdd/ReadVariableOp2L
$numeric_output/MatMul/ReadVariableOp$numeric_output/MatMul/ReadVariableOp2H
"quality_embedding/embedding_lookup"quality_embedding/embedding_lookup2N
%quality_output/BiasAdd/ReadVariableOp%quality_output/BiasAdd/ReadVariableOp2L
$quality_output/MatMul/ReadVariableOp$quality_output/MatMul/ReadVariableOp2D
 shift_embedding/embedding_lookup shift_embedding/embedding_lookup2J
#shift_output/BiasAdd/ReadVariableOp#shift_output/BiasAdd/ReadVariableOp2H
"shift_output/MatMul/ReadVariableOp"shift_output/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�	
�
L__inference_shift_embedding_layer_call_and_return_conditional_losses_1774453

inputs
embedding_lookup_1774447
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_1774447Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1774447*+
_output_shapes
:���������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1774447*+
_output_shapes
:���������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_dense_layer_call_and_return_conditional_losses_1773903

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�r
�

"__inference__wrapped_model_1773760
numeric_input
shift_input
quality_input<
8harmony_model_quality_embedding_embedding_lookup_1773704:
6harmony_model_shift_embedding_embedding_lookup_1773710<
8harmony_model_numeric_embedding_embedding_lookup_17737166
2harmony_model_dense_matmul_readvariableop_resource7
3harmony_model_dense_biasadd_readvariableop_resource?
;harmony_model_quality_output_matmul_readvariableop_resource@
<harmony_model_quality_output_biasadd_readvariableop_resource=
9harmony_model_shift_output_matmul_readvariableop_resource>
:harmony_model_shift_output_biasadd_readvariableop_resource?
;harmony_model_numeric_output_matmul_readvariableop_resource@
<harmony_model_numeric_output_biasadd_readvariableop_resource
identity

identity_1

identity_2��*harmony_model/dense/BiasAdd/ReadVariableOp�)harmony_model/dense/MatMul/ReadVariableOp�0harmony_model/numeric_embedding/embedding_lookup�3harmony_model/numeric_output/BiasAdd/ReadVariableOp�2harmony_model/numeric_output/MatMul/ReadVariableOp�0harmony_model/quality_embedding/embedding_lookup�3harmony_model/quality_output/BiasAdd/ReadVariableOp�2harmony_model/quality_output/MatMul/ReadVariableOp�.harmony_model/shift_embedding/embedding_lookup�1harmony_model/shift_output/BiasAdd/ReadVariableOp�0harmony_model/shift_output/MatMul/ReadVariableOp�
$harmony_model/quality_embedding/CastCastquality_input*

DstT0*

SrcT0*'
_output_shapes
:���������2&
$harmony_model/quality_embedding/Cast�
0harmony_model/quality_embedding/embedding_lookupResourceGather8harmony_model_quality_embedding_embedding_lookup_1773704(harmony_model/quality_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*K
_classA
?=loc:@harmony_model/quality_embedding/embedding_lookup/1773704*+
_output_shapes
:���������*
dtype022
0harmony_model/quality_embedding/embedding_lookup�
9harmony_model/quality_embedding/embedding_lookup/IdentityIdentity9harmony_model/quality_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@harmony_model/quality_embedding/embedding_lookup/1773704*+
_output_shapes
:���������2;
9harmony_model/quality_embedding/embedding_lookup/Identity�
;harmony_model/quality_embedding/embedding_lookup/Identity_1IdentityBharmony_model/quality_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2=
;harmony_model/quality_embedding/embedding_lookup/Identity_1�
"harmony_model/shift_embedding/CastCastshift_input*

DstT0*

SrcT0*'
_output_shapes
:���������2$
"harmony_model/shift_embedding/Cast�
.harmony_model/shift_embedding/embedding_lookupResourceGather6harmony_model_shift_embedding_embedding_lookup_1773710&harmony_model/shift_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*I
_class?
=;loc:@harmony_model/shift_embedding/embedding_lookup/1773710*+
_output_shapes
:���������*
dtype020
.harmony_model/shift_embedding/embedding_lookup�
7harmony_model/shift_embedding/embedding_lookup/IdentityIdentity7harmony_model/shift_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*I
_class?
=;loc:@harmony_model/shift_embedding/embedding_lookup/1773710*+
_output_shapes
:���������29
7harmony_model/shift_embedding/embedding_lookup/Identity�
9harmony_model/shift_embedding/embedding_lookup/Identity_1Identity@harmony_model/shift_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2;
9harmony_model/shift_embedding/embedding_lookup/Identity_1�
$harmony_model/numeric_embedding/CastCastnumeric_input*

DstT0*

SrcT0*'
_output_shapes
:���������2&
$harmony_model/numeric_embedding/Cast�
0harmony_model/numeric_embedding/embedding_lookupResourceGather8harmony_model_numeric_embedding_embedding_lookup_1773716(harmony_model/numeric_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*K
_classA
?=loc:@harmony_model/numeric_embedding/embedding_lookup/1773716*+
_output_shapes
:���������*
dtype022
0harmony_model/numeric_embedding/embedding_lookup�
9harmony_model/numeric_embedding/embedding_lookup/IdentityIdentity9harmony_model/numeric_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@harmony_model/numeric_embedding/embedding_lookup/1773716*+
_output_shapes
:���������2;
9harmony_model/numeric_embedding/embedding_lookup/Identity�
;harmony_model/numeric_embedding/embedding_lookup/Identity_1IdentityBharmony_model/numeric_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2=
;harmony_model/numeric_embedding/embedding_lookup/Identity_1�
harmony_model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
harmony_model/flatten/Const�
harmony_model/flatten/ReshapeReshapeDharmony_model/numeric_embedding/embedding_lookup/Identity_1:output:0$harmony_model/flatten/Const:output:0*
T0*'
_output_shapes
:���������2
harmony_model/flatten/Reshape�
harmony_model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
harmony_model/flatten_1/Const�
harmony_model/flatten_1/ReshapeReshapeBharmony_model/shift_embedding/embedding_lookup/Identity_1:output:0&harmony_model/flatten_1/Const:output:0*
T0*'
_output_shapes
:���������2!
harmony_model/flatten_1/Reshape�
harmony_model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
harmony_model/flatten_2/Const�
harmony_model/flatten_2/ReshapeReshapeDharmony_model/quality_embedding/embedding_lookup/Identity_1:output:0&harmony_model/flatten_2/Const:output:0*
T0*'
_output_shapes
:���������2!
harmony_model/flatten_2/Reshape�
%harmony_model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%harmony_model/concatenate/concat/axis�
 harmony_model/concatenate/concatConcatV2&harmony_model/flatten/Reshape:output:0(harmony_model/flatten_1/Reshape:output:0(harmony_model/flatten_2/Reshape:output:0.harmony_model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2"
 harmony_model/concatenate/concat�
)harmony_model/dense/MatMul/ReadVariableOpReadVariableOp2harmony_model_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02+
)harmony_model/dense/MatMul/ReadVariableOp�
harmony_model/dense/MatMulMatMul)harmony_model/concatenate/concat:output:01harmony_model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
harmony_model/dense/MatMul�
*harmony_model/dense/BiasAdd/ReadVariableOpReadVariableOp3harmony_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*harmony_model/dense/BiasAdd/ReadVariableOp�
harmony_model/dense/BiasAddBiasAdd$harmony_model/dense/MatMul:product:02harmony_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
harmony_model/dense/BiasAdd�
harmony_model/dense/ReluRelu$harmony_model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
harmony_model/dense/Relu�
2harmony_model/quality_output/MatMul/ReadVariableOpReadVariableOp;harmony_model_quality_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2harmony_model/quality_output/MatMul/ReadVariableOp�
#harmony_model/quality_output/MatMulMatMul&harmony_model/dense/Relu:activations:0:harmony_model/quality_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2%
#harmony_model/quality_output/MatMul�
3harmony_model/quality_output/BiasAdd/ReadVariableOpReadVariableOp<harmony_model_quality_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3harmony_model/quality_output/BiasAdd/ReadVariableOp�
$harmony_model/quality_output/BiasAddBiasAdd-harmony_model/quality_output/MatMul:product:0;harmony_model/quality_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2&
$harmony_model/quality_output/BiasAdd�
$harmony_model/quality_output/SoftmaxSoftmax-harmony_model/quality_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2&
$harmony_model/quality_output/Softmax�
0harmony_model/shift_output/MatMul/ReadVariableOpReadVariableOp9harmony_model_shift_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype022
0harmony_model/shift_output/MatMul/ReadVariableOp�
!harmony_model/shift_output/MatMulMatMul&harmony_model/dense/Relu:activations:08harmony_model/shift_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!harmony_model/shift_output/MatMul�
1harmony_model/shift_output/BiasAdd/ReadVariableOpReadVariableOp:harmony_model_shift_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1harmony_model/shift_output/BiasAdd/ReadVariableOp�
"harmony_model/shift_output/BiasAddBiasAdd+harmony_model/shift_output/MatMul:product:09harmony_model/shift_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"harmony_model/shift_output/BiasAdd�
"harmony_model/shift_output/SoftmaxSoftmax+harmony_model/shift_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2$
"harmony_model/shift_output/Softmax�
2harmony_model/numeric_output/MatMul/ReadVariableOpReadVariableOp;harmony_model_numeric_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype024
2harmony_model/numeric_output/MatMul/ReadVariableOp�
#harmony_model/numeric_output/MatMulMatMul&harmony_model/dense/Relu:activations:0:harmony_model/numeric_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2%
#harmony_model/numeric_output/MatMul�
3harmony_model/numeric_output/BiasAdd/ReadVariableOpReadVariableOp<harmony_model_numeric_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3harmony_model/numeric_output/BiasAdd/ReadVariableOp�
$harmony_model/numeric_output/BiasAddBiasAdd-harmony_model/numeric_output/MatMul:product:0;harmony_model/numeric_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2&
$harmony_model/numeric_output/BiasAdd�
$harmony_model/numeric_output/SoftmaxSoftmax-harmony_model/numeric_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2&
$harmony_model/numeric_output/Softmax�
IdentityIdentity.harmony_model/numeric_output/Softmax:softmax:0+^harmony_model/dense/BiasAdd/ReadVariableOp*^harmony_model/dense/MatMul/ReadVariableOp1^harmony_model/numeric_embedding/embedding_lookup4^harmony_model/numeric_output/BiasAdd/ReadVariableOp3^harmony_model/numeric_output/MatMul/ReadVariableOp1^harmony_model/quality_embedding/embedding_lookup4^harmony_model/quality_output/BiasAdd/ReadVariableOp3^harmony_model/quality_output/MatMul/ReadVariableOp/^harmony_model/shift_embedding/embedding_lookup2^harmony_model/shift_output/BiasAdd/ReadVariableOp1^harmony_model/shift_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity.harmony_model/quality_output/Softmax:softmax:0+^harmony_model/dense/BiasAdd/ReadVariableOp*^harmony_model/dense/MatMul/ReadVariableOp1^harmony_model/numeric_embedding/embedding_lookup4^harmony_model/numeric_output/BiasAdd/ReadVariableOp3^harmony_model/numeric_output/MatMul/ReadVariableOp1^harmony_model/quality_embedding/embedding_lookup4^harmony_model/quality_output/BiasAdd/ReadVariableOp3^harmony_model/quality_output/MatMul/ReadVariableOp/^harmony_model/shift_embedding/embedding_lookup2^harmony_model/shift_output/BiasAdd/ReadVariableOp1^harmony_model/shift_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity,harmony_model/shift_output/Softmax:softmax:0+^harmony_model/dense/BiasAdd/ReadVariableOp*^harmony_model/dense/MatMul/ReadVariableOp1^harmony_model/numeric_embedding/embedding_lookup4^harmony_model/numeric_output/BiasAdd/ReadVariableOp3^harmony_model/numeric_output/MatMul/ReadVariableOp1^harmony_model/quality_embedding/embedding_lookup4^harmony_model/quality_output/BiasAdd/ReadVariableOp3^harmony_model/quality_output/MatMul/ReadVariableOp/^harmony_model/shift_embedding/embedding_lookup2^harmony_model/shift_output/BiasAdd/ReadVariableOp1^harmony_model/shift_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::2X
*harmony_model/dense/BiasAdd/ReadVariableOp*harmony_model/dense/BiasAdd/ReadVariableOp2V
)harmony_model/dense/MatMul/ReadVariableOp)harmony_model/dense/MatMul/ReadVariableOp2d
0harmony_model/numeric_embedding/embedding_lookup0harmony_model/numeric_embedding/embedding_lookup2j
3harmony_model/numeric_output/BiasAdd/ReadVariableOp3harmony_model/numeric_output/BiasAdd/ReadVariableOp2h
2harmony_model/numeric_output/MatMul/ReadVariableOp2harmony_model/numeric_output/MatMul/ReadVariableOp2d
0harmony_model/quality_embedding/embedding_lookup0harmony_model/quality_embedding/embedding_lookup2j
3harmony_model/quality_output/BiasAdd/ReadVariableOp3harmony_model/quality_output/BiasAdd/ReadVariableOp2h
2harmony_model/quality_output/MatMul/ReadVariableOp2harmony_model/quality_output/MatMul/ReadVariableOp2`
.harmony_model/shift_embedding/embedding_lookup.harmony_model/shift_embedding/embedding_lookup2f
1harmony_model/shift_output/BiasAdd/ReadVariableOp1harmony_model/shift_output/BiasAdd/ReadVariableOp2d
0harmony_model/shift_output/MatMul/ReadVariableOp0harmony_model/shift_output/MatMul/ReadVariableOp:V R
'
_output_shapes
:���������
'
_user_specified_namenumeric_input:TP
'
_output_shapes
:���������
%
_user_specified_nameshift_input:VR
'
_output_shapes
:���������
'
_user_specified_namequality_input
�
g
-__inference_concatenate_layer_call_fn_1774525
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_17738822
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�
�
.__inference_shift_output_layer_call_fn_1774585

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_shift_output_layer_call_and_return_conditional_losses_17739572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1774236
numeric_input
quality_input
shift_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnumeric_inputshift_inputquality_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_17737602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namenumeric_input:VR
'
_output_shapes
:���������
'
_user_specified_namequality_input:TP
'
_output_shapes
:���������
%
_user_specified_nameshift_input
�	
�
N__inference_quality_embedding_layer_call_and_return_conditional_losses_1774470

inputs
embedding_lookup_1774464
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_1774464Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1774464*+
_output_shapes
:���������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1774464*+
_output_shapes
:���������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
G
+__inference_flatten_1_layer_call_fn_1774499

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_17738522
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
K__inference_numeric_output_layer_call_and_return_conditional_losses_1774556

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_1774954
file_prefix1
-assignvariableop_numeric_embedding_embeddings1
-assignvariableop_1_shift_embedding_embeddings3
/assignvariableop_2_quality_embedding_embeddings#
assignvariableop_3_dense_kernel!
assignvariableop_4_dense_bias,
(assignvariableop_5_numeric_output_kernel*
&assignvariableop_6_numeric_output_bias*
&assignvariableop_7_shift_output_kernel(
$assignvariableop_8_shift_output_bias,
(assignvariableop_9_quality_output_kernel+
'assignvariableop_10_quality_output_bias!
assignvariableop_11_adam_iter#
assignvariableop_12_adam_beta_1#
assignvariableop_13_adam_beta_2"
assignvariableop_14_adam_decay*
&assignvariableop_15_adam_learning_rate
assignvariableop_16_total
assignvariableop_17_count
assignvariableop_18_total_1
assignvariableop_19_count_1
assignvariableop_20_total_2
assignvariableop_21_count_2
assignvariableop_22_total_3
assignvariableop_23_count_3
assignvariableop_24_total_4
assignvariableop_25_count_4
assignvariableop_26_total_5
assignvariableop_27_count_5
assignvariableop_28_total_6
assignvariableop_29_count_6;
7assignvariableop_30_adam_numeric_embedding_embeddings_m9
5assignvariableop_31_adam_shift_embedding_embeddings_m;
7assignvariableop_32_adam_quality_embedding_embeddings_m+
'assignvariableop_33_adam_dense_kernel_m)
%assignvariableop_34_adam_dense_bias_m4
0assignvariableop_35_adam_numeric_output_kernel_m2
.assignvariableop_36_adam_numeric_output_bias_m2
.assignvariableop_37_adam_shift_output_kernel_m0
,assignvariableop_38_adam_shift_output_bias_m4
0assignvariableop_39_adam_quality_output_kernel_m2
.assignvariableop_40_adam_quality_output_bias_m;
7assignvariableop_41_adam_numeric_embedding_embeddings_v9
5assignvariableop_42_adam_shift_embedding_embeddings_v;
7assignvariableop_43_adam_quality_embedding_embeddings_v+
'assignvariableop_44_adam_dense_kernel_v)
%assignvariableop_45_adam_dense_bias_v4
0assignvariableop_46_adam_numeric_output_kernel_v2
.assignvariableop_47_adam_numeric_output_bias_v2
.assignvariableop_48_adam_shift_output_kernel_v0
,assignvariableop_49_adam_shift_output_bias_v4
0assignvariableop_50_adam_quality_output_kernel_v2
.assignvariableop_51_adam_quality_output_bias_v
identity_53��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*�
value�B�5B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp-assignvariableop_numeric_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp-assignvariableop_1_shift_embedding_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_quality_embedding_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp(assignvariableop_5_numeric_output_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_numeric_output_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp&assignvariableop_7_shift_output_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp$assignvariableop_8_shift_output_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp(assignvariableop_9_quality_output_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_quality_output_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_3Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_3Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_4Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_4Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_5Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_5Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpassignvariableop_28_total_6Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_6Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp7assignvariableop_30_adam_numeric_embedding_embeddings_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp5assignvariableop_31_adam_shift_embedding_embeddings_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp7assignvariableop_32_adam_quality_embedding_embeddings_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dense_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_dense_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp0assignvariableop_35_adam_numeric_output_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp.assignvariableop_36_adam_numeric_output_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp.assignvariableop_37_adam_shift_output_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_shift_output_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_quality_output_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp.assignvariableop_40_adam_quality_output_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_numeric_embedding_embeddings_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_shift_embedding_embeddings_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_quality_embedding_embeddings_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp%assignvariableop_45_adam_dense_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp0assignvariableop_46_adam_numeric_output_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp.assignvariableop_47_adam_numeric_output_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp.assignvariableop_48_adam_shift_output_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adam_shift_output_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp0assignvariableop_50_adam_quality_output_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp.assignvariableop_51_adam_quality_output_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_519
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_52�	
Identity_53IdentityIdentity_52:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_53"#
identity_53Identity_53:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
�	
�
L__inference_shift_embedding_layer_call_and_return_conditional_losses_1773798

inputs
embedding_lookup_1773792
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_1773792Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1773792*+
_output_shapes
:���������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1773792*+
_output_shapes
:���������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1774494

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_concatenate_layer_call_and_return_conditional_losses_1774518
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�
y
3__inference_quality_embedding_layer_call_fn_1774477

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_quality_embedding_layer_call_and_return_conditional_losses_17737762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
E
)__inference_flatten_layer_call_fn_1774488

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17738382
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_1773866

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_1774483

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
N__inference_quality_embedding_layer_call_and_return_conditional_losses_1773776

inputs
embedding_lookup_1773770
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_1773770Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1773770*+
_output_shapes
:���������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1773770*+
_output_shapes
:���������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_harmony_model_layer_call_fn_1774426
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_harmony_model_layer_call_and_return_conditional_losses_17741642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�
G
+__inference_flatten_2_layer_call_fn_1774510

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_17738662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_1773838

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�<
�
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774164

inputs
inputs_1
inputs_2
quality_embedding_1774128
shift_embedding_1774131
numeric_embedding_1774134
dense_1774141
dense_1774143
quality_output_1774146
quality_output_1774148
shift_output_1774151
shift_output_1774153
numeric_output_1774156
numeric_output_1774158
identity

identity_1

identity_2��dense/StatefulPartitionedCall�)numeric_embedding/StatefulPartitionedCall�&numeric_output/StatefulPartitionedCall�)quality_embedding/StatefulPartitionedCall�&quality_output/StatefulPartitionedCall�'shift_embedding/StatefulPartitionedCall�$shift_output/StatefulPartitionedCall�
)quality_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_2quality_embedding_1774128*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_quality_embedding_layer_call_and_return_conditional_losses_17737762+
)quality_embedding/StatefulPartitionedCall�
'shift_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1shift_embedding_1774131*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_shift_embedding_layer_call_and_return_conditional_losses_17737982)
'shift_embedding/StatefulPartitionedCall�
)numeric_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsnumeric_embedding_1774134*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_numeric_embedding_layer_call_and_return_conditional_losses_17738202+
)numeric_embedding/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall2numeric_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17738382
flatten/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall0shift_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_17738522
flatten_1/PartitionedCall�
flatten_2/PartitionedCallPartitionedCall2quality_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_17738662
flatten_2/PartitionedCall�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_17738822
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1774141dense_1774143*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_17739032
dense/StatefulPartitionedCall�
&quality_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0quality_output_1774146quality_output_1774148*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quality_output_layer_call_and_return_conditional_losses_17739302(
&quality_output/StatefulPartitionedCall�
$shift_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0shift_output_1774151shift_output_1774153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_shift_output_layer_call_and_return_conditional_losses_17739572&
$shift_output/StatefulPartitionedCall�
&numeric_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0numeric_output_1774156numeric_output_1774158*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_numeric_output_layer_call_and_return_conditional_losses_17739842(
&numeric_output/StatefulPartitionedCall�
IdentityIdentity/numeric_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity-shift_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity/quality_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2V
)numeric_embedding/StatefulPartitionedCall)numeric_embedding/StatefulPartitionedCall2P
&numeric_output/StatefulPartitionedCall&numeric_output/StatefulPartitionedCall2V
)quality_embedding/StatefulPartitionedCall)quality_embedding/StatefulPartitionedCall2P
&quality_output/StatefulPartitionedCall&quality_output/StatefulPartitionedCall2R
'shift_embedding/StatefulPartitionedCall'shift_embedding/StatefulPartitionedCall2L
$shift_output/StatefulPartitionedCall$shift_output/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�=
�
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774003
numeric_input
shift_input
quality_input
quality_embedding_1773785
shift_embedding_1773807
numeric_embedding_1773829
dense_1773914
dense_1773916
quality_output_1773941
quality_output_1773943
shift_output_1773968
shift_output_1773970
numeric_output_1773995
numeric_output_1773997
identity

identity_1

identity_2��dense/StatefulPartitionedCall�)numeric_embedding/StatefulPartitionedCall�&numeric_output/StatefulPartitionedCall�)quality_embedding/StatefulPartitionedCall�&quality_output/StatefulPartitionedCall�'shift_embedding/StatefulPartitionedCall�$shift_output/StatefulPartitionedCall�
)quality_embedding/StatefulPartitionedCallStatefulPartitionedCallquality_inputquality_embedding_1773785*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_quality_embedding_layer_call_and_return_conditional_losses_17737762+
)quality_embedding/StatefulPartitionedCall�
'shift_embedding/StatefulPartitionedCallStatefulPartitionedCallshift_inputshift_embedding_1773807*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_shift_embedding_layer_call_and_return_conditional_losses_17737982)
'shift_embedding/StatefulPartitionedCall�
)numeric_embedding/StatefulPartitionedCallStatefulPartitionedCallnumeric_inputnumeric_embedding_1773829*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_numeric_embedding_layer_call_and_return_conditional_losses_17738202+
)numeric_embedding/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall2numeric_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17738382
flatten/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall0shift_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_17738522
flatten_1/PartitionedCall�
flatten_2/PartitionedCallPartitionedCall2quality_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_17738662
flatten_2/PartitionedCall�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_17738822
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1773914dense_1773916*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_17739032
dense/StatefulPartitionedCall�
&quality_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0quality_output_1773941quality_output_1773943*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quality_output_layer_call_and_return_conditional_losses_17739302(
&quality_output/StatefulPartitionedCall�
$shift_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0shift_output_1773968shift_output_1773970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_shift_output_layer_call_and_return_conditional_losses_17739572&
$shift_output/StatefulPartitionedCall�
&numeric_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0numeric_output_1773995numeric_output_1773997*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_numeric_output_layer_call_and_return_conditional_losses_17739842(
&numeric_output/StatefulPartitionedCall�
IdentityIdentity/numeric_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity-shift_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity/quality_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2V
)numeric_embedding/StatefulPartitionedCall)numeric_embedding/StatefulPartitionedCall2P
&numeric_output/StatefulPartitionedCall&numeric_output/StatefulPartitionedCall2V
)quality_embedding/StatefulPartitionedCall)quality_embedding/StatefulPartitionedCall2P
&quality_output/StatefulPartitionedCall&quality_output/StatefulPartitionedCall2R
'shift_embedding/StatefulPartitionedCall'shift_embedding/StatefulPartitionedCall2L
$shift_output/StatefulPartitionedCall$shift_output/StatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namenumeric_input:TP
'
_output_shapes
:���������
%
_user_specified_nameshift_input:VR
'
_output_shapes
:���������
'
_user_specified_namequality_input
�	
�
N__inference_numeric_embedding_layer_call_and_return_conditional_losses_1773820

inputs
embedding_lookup_1773814
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_1773814Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1773814*+
_output_shapes
:���������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1773814*+
_output_shapes
:���������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�=
�
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774044
numeric_input
shift_input
quality_input
quality_embedding_1774008
shift_embedding_1774011
numeric_embedding_1774014
dense_1774021
dense_1774023
quality_output_1774026
quality_output_1774028
shift_output_1774031
shift_output_1774033
numeric_output_1774036
numeric_output_1774038
identity

identity_1

identity_2��dense/StatefulPartitionedCall�)numeric_embedding/StatefulPartitionedCall�&numeric_output/StatefulPartitionedCall�)quality_embedding/StatefulPartitionedCall�&quality_output/StatefulPartitionedCall�'shift_embedding/StatefulPartitionedCall�$shift_output/StatefulPartitionedCall�
)quality_embedding/StatefulPartitionedCallStatefulPartitionedCallquality_inputquality_embedding_1774008*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_quality_embedding_layer_call_and_return_conditional_losses_17737762+
)quality_embedding/StatefulPartitionedCall�
'shift_embedding/StatefulPartitionedCallStatefulPartitionedCallshift_inputshift_embedding_1774011*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_shift_embedding_layer_call_and_return_conditional_losses_17737982)
'shift_embedding/StatefulPartitionedCall�
)numeric_embedding/StatefulPartitionedCallStatefulPartitionedCallnumeric_inputnumeric_embedding_1774014*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_numeric_embedding_layer_call_and_return_conditional_losses_17738202+
)numeric_embedding/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall2numeric_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17738382
flatten/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall0shift_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_17738522
flatten_1/PartitionedCall�
flatten_2/PartitionedCallPartitionedCall2quality_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_17738662
flatten_2/PartitionedCall�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_17738822
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1774021dense_1774023*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_17739032
dense/StatefulPartitionedCall�
&quality_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0quality_output_1774026quality_output_1774028*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_quality_output_layer_call_and_return_conditional_losses_17739302(
&quality_output/StatefulPartitionedCall�
$shift_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0shift_output_1774031shift_output_1774033*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_shift_output_layer_call_and_return_conditional_losses_17739572&
$shift_output/StatefulPartitionedCall�
&numeric_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0numeric_output_1774036numeric_output_1774038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_numeric_output_layer_call_and_return_conditional_losses_17739842(
&numeric_output/StatefulPartitionedCall�
IdentityIdentity/numeric_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity-shift_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity/quality_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*^numeric_embedding/StatefulPartitionedCall'^numeric_output/StatefulPartitionedCall*^quality_embedding/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall(^shift_embedding/StatefulPartitionedCall%^shift_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2V
)numeric_embedding/StatefulPartitionedCall)numeric_embedding/StatefulPartitionedCall2P
&numeric_output/StatefulPartitionedCall&numeric_output/StatefulPartitionedCall2V
)quality_embedding/StatefulPartitionedCall)quality_embedding/StatefulPartitionedCall2P
&quality_output/StatefulPartitionedCall&quality_output/StatefulPartitionedCall2R
'shift_embedding/StatefulPartitionedCall'shift_embedding/StatefulPartitionedCall2L
$shift_output/StatefulPartitionedCall$shift_output/StatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namenumeric_input:TP
'
_output_shapes
:���������
%
_user_specified_nameshift_input:VR
'
_output_shapes
:���������
'
_user_specified_namequality_input
�	
�
N__inference_numeric_embedding_layer_call_and_return_conditional_losses_1774436

inputs
embedding_lookup_1774430
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_1774430Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1774430*+
_output_shapes
:���������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1774430*+
_output_shapes
:���������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_1774505

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_harmony_model_layer_call_fn_1774193
numeric_input
shift_input
quality_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnumeric_inputshift_inputquality_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_harmony_model_layer_call_and_return_conditional_losses_17741642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namenumeric_input:TP
'
_output_shapes
:���������
%
_user_specified_nameshift_input:VR
'
_output_shapes
:���������
'
_user_specified_namequality_input
�
y
3__inference_numeric_embedding_layer_call_fn_1774443

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_numeric_embedding_layer_call_and_return_conditional_losses_17738202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
K__inference_quality_output_layer_call_and_return_conditional_losses_1773930

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
B__inference_dense_layer_call_and_return_conditional_losses_1774536

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
|
'__inference_dense_layer_call_fn_1774545

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_17739032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_shift_output_layer_call_and_return_conditional_losses_1774576

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�g
�
 __inference__traced_save_1774788
file_prefix;
7savev2_numeric_embedding_embeddings_read_readvariableop9
5savev2_shift_embedding_embeddings_read_readvariableop;
7savev2_quality_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop4
0savev2_numeric_output_kernel_read_readvariableop2
.savev2_numeric_output_bias_read_readvariableop2
.savev2_shift_output_kernel_read_readvariableop0
,savev2_shift_output_bias_read_readvariableop4
0savev2_quality_output_kernel_read_readvariableop2
.savev2_quality_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_5_read_readvariableop&
"savev2_count_5_read_readvariableop&
"savev2_total_6_read_readvariableop&
"savev2_count_6_read_readvariableopB
>savev2_adam_numeric_embedding_embeddings_m_read_readvariableop@
<savev2_adam_shift_embedding_embeddings_m_read_readvariableopB
>savev2_adam_quality_embedding_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop;
7savev2_adam_numeric_output_kernel_m_read_readvariableop9
5savev2_adam_numeric_output_bias_m_read_readvariableop9
5savev2_adam_shift_output_kernel_m_read_readvariableop7
3savev2_adam_shift_output_bias_m_read_readvariableop;
7savev2_adam_quality_output_kernel_m_read_readvariableop9
5savev2_adam_quality_output_bias_m_read_readvariableopB
>savev2_adam_numeric_embedding_embeddings_v_read_readvariableop@
<savev2_adam_shift_embedding_embeddings_v_read_readvariableopB
>savev2_adam_quality_embedding_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop;
7savev2_adam_numeric_output_kernel_v_read_readvariableop9
5savev2_adam_numeric_output_bias_v_read_readvariableop9
5savev2_adam_shift_output_kernel_v_read_readvariableop7
3savev2_adam_shift_output_bias_v_read_readvariableop;
7savev2_adam_quality_output_kernel_v_read_readvariableop9
5savev2_adam_quality_output_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*�
value�B�5B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_numeric_embedding_embeddings_read_readvariableop5savev2_shift_embedding_embeddings_read_readvariableop7savev2_quality_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop0savev2_numeric_output_kernel_read_readvariableop.savev2_numeric_output_bias_read_readvariableop.savev2_shift_output_kernel_read_readvariableop,savev2_shift_output_bias_read_readvariableop0savev2_quality_output_kernel_read_readvariableop.savev2_quality_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_5_read_readvariableop"savev2_count_5_read_readvariableop"savev2_total_6_read_readvariableop"savev2_count_6_read_readvariableop>savev2_adam_numeric_embedding_embeddings_m_read_readvariableop<savev2_adam_shift_embedding_embeddings_m_read_readvariableop>savev2_adam_quality_embedding_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop7savev2_adam_numeric_output_kernel_m_read_readvariableop5savev2_adam_numeric_output_bias_m_read_readvariableop5savev2_adam_shift_output_kernel_m_read_readvariableop3savev2_adam_shift_output_bias_m_read_readvariableop7savev2_adam_quality_output_kernel_m_read_readvariableop5savev2_adam_quality_output_bias_m_read_readvariableop>savev2_adam_numeric_embedding_embeddings_v_read_readvariableop<savev2_adam_shift_embedding_embeddings_v_read_readvariableop>savev2_adam_quality_embedding_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop7savev2_adam_numeric_output_kernel_v_read_readvariableop5savev2_adam_numeric_output_bias_v_read_readvariableop5savev2_adam_shift_output_kernel_v_read_readvariableop3savev2_adam_shift_output_bias_v_read_readvariableop7savev2_adam_quality_output_kernel_v_read_readvariableop5savev2_adam_quality_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *C
dtypes9
725	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::	�:�:	�::	�::	�:: : : : : : : : : : : : : : : : : : : ::::	�:�:	�::	�::	�:::::	�:�:	�::	�::	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 	

_output_shapes
::%
!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$  

_output_shapes

::$! 

_output_shapes

::%"!

_output_shapes
:	�:!#

_output_shapes	
:�:%$!

_output_shapes
:	�: %

_output_shapes
::%&!

_output_shapes
:	�: '

_output_shapes
::%(!

_output_shapes
:	�: )

_output_shapes
::$* 

_output_shapes

::$+ 

_output_shapes

::$, 

_output_shapes

::%-!

_output_shapes
:	�:!.

_output_shapes	
:�:%/!

_output_shapes
:	�: 0

_output_shapes
::%1!

_output_shapes
:	�: 2

_output_shapes
::%3!

_output_shapes
:	�: 4

_output_shapes
::5

_output_shapes
: 
�
�
H__inference_concatenate_layer_call_and_return_conditional_losses_1773882

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:���������:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
K__inference_numeric_output_layer_call_and_return_conditional_losses_1773984

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
I__inference_shift_output_layer_call_and_return_conditional_losses_1773957

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_harmony_model_layer_call_fn_1774119
numeric_input
shift_input
quality_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnumeric_inputshift_inputquality_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_harmony_model_layer_call_and_return_conditional_losses_17740902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*x
_input_shapesg
e:���������:���������:���������:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namenumeric_input:TP
'
_output_shapes
:���������
%
_user_specified_nameshift_input:VR
'
_output_shapes
:���������
'
_user_specified_namequality_input
�

�
K__inference_quality_output_layer_call_and_return_conditional_losses_1774596

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
0__inference_numeric_output_layer_call_fn_1774565

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_numeric_output_layer_call_and_return_conditional_losses_17739842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
numeric_input6
serving_default_numeric_input:0���������
G
quality_input6
serving_default_quality_input:0���������
C
shift_input4
serving_default_shift_input:0���������B
numeric_output0
StatefulPartitionedCall:0���������B
quality_output0
StatefulPartitionedCall:1���������@
shift_output0
StatefulPartitionedCall:2���������tensorflow/serving/predict:��
�g
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"�c
_tf_keras_network�c{"class_name": "Functional", "name": "harmony_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "harmony_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "numeric_input"}, "name": "numeric_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "shift_input"}, "name": "shift_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "quality_input"}, "name": "quality_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "numeric_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 7, "output_dim": 2, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "numeric_embedding", "inbound_nodes": [[["numeric_input", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "shift_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3, "output_dim": 2, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "shift_embedding", "inbound_nodes": [[["shift_input", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "quality_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 15, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "quality_embedding", "inbound_nodes": [[["quality_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["numeric_embedding", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["shift_embedding", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["quality_embedding", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["flatten", 0, 0, {}], ["flatten_1", 0, 0, {}], ["flatten_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "numeric_output", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "numeric_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "shift_output", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "shift_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "quality_output", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "quality_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["numeric_input", 0, 0], ["shift_input", 0, 0], ["quality_input", 0, 0]], "output_layers": [["numeric_output", 0, 0], ["shift_output", 0, 0], ["quality_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "harmony_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "numeric_input"}, "name": "numeric_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "shift_input"}, "name": "shift_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "quality_input"}, "name": "quality_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "numeric_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 7, "output_dim": 2, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "numeric_embedding", "inbound_nodes": [[["numeric_input", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "shift_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3, "output_dim": 2, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "shift_embedding", "inbound_nodes": [[["shift_input", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "quality_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 15, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "quality_embedding", "inbound_nodes": [[["quality_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["numeric_embedding", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["shift_embedding", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["quality_embedding", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["flatten", 0, 0, {}], ["flatten_1", 0, 0, {}], ["flatten_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "numeric_output", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "numeric_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "shift_output", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "shift_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "quality_output", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "quality_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["numeric_input", 0, 0], ["shift_input", 0, 0], ["quality_input", 0, 0]], "output_layers": [["numeric_output", 0, 0], ["shift_output", 0, 0], ["quality_output", 0, 0]]}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "numeric_output_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}], [{"class_name": "MeanMetricWrapper", "config": {"name": "shift_output_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}], [{"class_name": "MeanMetricWrapper", "config": {"name": "quality_output_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "numeric_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "numeric_input"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "shift_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "shift_input"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "quality_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "quality_input"}}
�

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "numeric_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "numeric_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 7, "output_dim": 2, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
�

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "shift_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "shift_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3, "output_dim": 2, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
�

embeddings
 trainable_variables
!	variables
"regularization_losses
#	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "quality_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "quality_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 15, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
�
$trainable_variables
%	variables
&regularization_losses
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
(trainable_variables
)	variables
*regularization_losses
+	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
,trainable_variables
-	variables
.regularization_losses
/	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
0trainable_variables
1	variables
2regularization_losses
3	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 4]}]}
�

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
�

:kernel
;bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "numeric_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "numeric_output", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
�

@kernel
Abias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "shift_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "shift_output", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
�

Fkernel
Gbias
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "quality_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quality_output", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
�
Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_ratem�m�m�4m�5m�:m�;m�@m�Am�Fm�Gm�v�v�v�4v�5v�:v�;v�@v�Av�Fv�Gv�"
	optimizer
n
0
1
2
43
54
:5
;6
@7
A8
F9
G10"
trackable_list_wrapper
n
0
1
2
43
54
:5
;6
@7
A8
F9
G10"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Qmetrics
trainable_variables
Rlayer_regularization_losses
	variables

Slayers
regularization_losses
Tlayer_metrics
Unon_trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
.:,2numeric_embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Vmetrics
trainable_variables
Wlayer_regularization_losses
	variables

Xlayers
regularization_losses
Ylayer_metrics
Znon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*2shift_embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[metrics
trainable_variables
\layer_regularization_losses
	variables

]layers
regularization_losses
^layer_metrics
_non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2quality_embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
`metrics
 trainable_variables
alayer_regularization_losses
!	variables

blayers
"regularization_losses
clayer_metrics
dnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
emetrics
$trainable_variables
flayer_regularization_losses
%	variables

glayers
&regularization_losses
hlayer_metrics
inon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
jmetrics
(trainable_variables
klayer_regularization_losses
)	variables

llayers
*regularization_losses
mlayer_metrics
nnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
ometrics
,trainable_variables
player_regularization_losses
-	variables

qlayers
.regularization_losses
rlayer_metrics
snon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
tmetrics
0trainable_variables
ulayer_regularization_losses
1	variables

vlayers
2regularization_losses
wlayer_metrics
xnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	�2dense/kernel
:�2
dense/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ymetrics
6trainable_variables
zlayer_regularization_losses
7	variables

{layers
8regularization_losses
|layer_metrics
}non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&	�2numeric_output/kernel
!:2numeric_output/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
~metrics
<trainable_variables
layer_regularization_losses
=	variables
�layers
>regularization_losses
�layer_metrics
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$	�2shift_output/kernel
:2shift_output/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
Btrainable_variables
 �layer_regularization_losses
C	variables
�layers
Dregularization_losses
�layer_metrics
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&	�2quality_output/kernel
!:2quality_output/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
Htrainable_variables
 �layer_regularization_losses
I	variables
�layers
Jregularization_losses
�layer_metrics
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
 "
trackable_list_wrapper
�
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
13"
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
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "numeric_output_loss", "dtype": "float32", "config": {"name": "numeric_output_loss", "dtype": "float32"}}
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "shift_output_loss", "dtype": "float32", "config": {"name": "shift_output_loss", "dtype": "float32"}}
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "quality_output_loss", "dtype": "float32", "config": {"name": "quality_output_loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "numeric_output_accuracy", "dtype": "float32", "config": {"name": "numeric_output_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "shift_output_accuracy", "dtype": "float32", "config": {"name": "shift_output_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "quality_output_accuracy", "dtype": "float32", "config": {"name": "quality_output_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
3:12#Adam/numeric_embedding/embeddings/m
1:/2!Adam/shift_embedding/embeddings/m
3:12#Adam/quality_embedding/embeddings/m
$:"	�2Adam/dense/kernel/m
:�2Adam/dense/bias/m
-:+	�2Adam/numeric_output/kernel/m
&:$2Adam/numeric_output/bias/m
+:)	�2Adam/shift_output/kernel/m
$:"2Adam/shift_output/bias/m
-:+	�2Adam/quality_output/kernel/m
&:$2Adam/quality_output/bias/m
3:12#Adam/numeric_embedding/embeddings/v
1:/2!Adam/shift_embedding/embeddings/v
3:12#Adam/quality_embedding/embeddings/v
$:"	�2Adam/dense/kernel/v
:�2Adam/dense/bias/v
-:+	�2Adam/numeric_output/kernel/v
&:$2Adam/numeric_output/bias/v
+:)	�2Adam/shift_output/kernel/v
$:"2Adam/shift_output/bias/v
-:+	�2Adam/quality_output/kernel/v
&:$2Adam/quality_output/bias/v
�2�
/__inference_harmony_model_layer_call_fn_1774393
/__inference_harmony_model_layer_call_fn_1774193
/__inference_harmony_model_layer_call_fn_1774119
/__inference_harmony_model_layer_call_fn_1774426�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774360
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774044
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774298
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774003�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_1773760�
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
annotations� *��~
|�y
'�$
numeric_input���������
%�"
shift_input���������
'�$
quality_input���������
�2�
3__inference_numeric_embedding_layer_call_fn_1774443�
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
N__inference_numeric_embedding_layer_call_and_return_conditional_losses_1774436�
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
1__inference_shift_embedding_layer_call_fn_1774460�
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
L__inference_shift_embedding_layer_call_and_return_conditional_losses_1774453�
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
3__inference_quality_embedding_layer_call_fn_1774477�
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
N__inference_quality_embedding_layer_call_and_return_conditional_losses_1774470�
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
)__inference_flatten_layer_call_fn_1774488�
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
D__inference_flatten_layer_call_and_return_conditional_losses_1774483�
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
+__inference_flatten_1_layer_call_fn_1774499�
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
F__inference_flatten_1_layer_call_and_return_conditional_losses_1774494�
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
+__inference_flatten_2_layer_call_fn_1774510�
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
F__inference_flatten_2_layer_call_and_return_conditional_losses_1774505�
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
-__inference_concatenate_layer_call_fn_1774525�
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
H__inference_concatenate_layer_call_and_return_conditional_losses_1774518�
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
'__inference_dense_layer_call_fn_1774545�
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
B__inference_dense_layer_call_and_return_conditional_losses_1774536�
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
0__inference_numeric_output_layer_call_fn_1774565�
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
K__inference_numeric_output_layer_call_and_return_conditional_losses_1774556�
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
.__inference_shift_output_layer_call_fn_1774585�
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
I__inference_shift_output_layer_call_and_return_conditional_losses_1774576�
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
0__inference_quality_output_layer_call_fn_1774605�
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
K__inference_quality_output_layer_call_and_return_conditional_losses_1774596�
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
�B�
%__inference_signature_wrapper_1774236numeric_inputquality_inputshift_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_1773760�45FG@A:;���
��~
|�y
'�$
numeric_input���������
%�"
shift_input���������
'�$
quality_input���������
� "���
:
numeric_output(�%
numeric_output���������
:
quality_output(�%
quality_output���������
6
shift_output&�#
shift_output����������
H__inference_concatenate_layer_call_and_return_conditional_losses_1774518�~�{
t�q
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
� "%�"
�
0���������
� �
-__inference_concatenate_layer_call_fn_1774525�~�{
t�q
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
� "�����������
B__inference_dense_layer_call_and_return_conditional_losses_1774536]45/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� {
'__inference_dense_layer_call_fn_1774545P45/�,
%�"
 �
inputs���������
� "������������
F__inference_flatten_1_layer_call_and_return_conditional_losses_1774494\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� ~
+__inference_flatten_1_layer_call_fn_1774499O3�0
)�&
$�!
inputs���������
� "�����������
F__inference_flatten_2_layer_call_and_return_conditional_losses_1774505\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� ~
+__inference_flatten_2_layer_call_fn_1774510O3�0
)�&
$�!
inputs���������
� "�����������
D__inference_flatten_layer_call_and_return_conditional_losses_1774483\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� |
)__inference_flatten_layer_call_fn_1774488O3�0
)�&
$�!
inputs���������
� "�����������
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774003�45FG@A:;���
���
|�y
'�$
numeric_input���������
%�"
shift_input���������
'�$
quality_input���������
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774044�45FG@A:;���
���
|�y
'�$
numeric_input���������
%�"
shift_input���������
'�$
quality_input���������
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774298�45FG@A:;���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
J__inference_harmony_model_layer_call_and_return_conditional_losses_1774360�45FG@A:;���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
/__inference_harmony_model_layer_call_fn_1774119�45FG@A:;���
���
|�y
'�$
numeric_input���������
%�"
shift_input���������
'�$
quality_input���������
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
/__inference_harmony_model_layer_call_fn_1774193�45FG@A:;���
���
|�y
'�$
numeric_input���������
%�"
shift_input���������
'�$
quality_input���������
p 

 
� "Z�W
�
0���������
�
1���������
�
2����������
/__inference_harmony_model_layer_call_fn_1774393�45FG@A:;���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
/__inference_harmony_model_layer_call_fn_1774426�45FG@A:;���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p 

 
� "Z�W
�
0���������
�
1���������
�
2����������
N__inference_numeric_embedding_layer_call_and_return_conditional_losses_1774436_/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� �
3__inference_numeric_embedding_layer_call_fn_1774443R/�,
%�"
 �
inputs���������
� "�����������
K__inference_numeric_output_layer_call_and_return_conditional_losses_1774556]:;0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
0__inference_numeric_output_layer_call_fn_1774565P:;0�-
&�#
!�
inputs����������
� "�����������
N__inference_quality_embedding_layer_call_and_return_conditional_losses_1774470_/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� �
3__inference_quality_embedding_layer_call_fn_1774477R/�,
%�"
 �
inputs���������
� "�����������
K__inference_quality_output_layer_call_and_return_conditional_losses_1774596]FG0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
0__inference_quality_output_layer_call_fn_1774605PFG0�-
&�#
!�
inputs����������
� "�����������
L__inference_shift_embedding_layer_call_and_return_conditional_losses_1774453_/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� �
1__inference_shift_embedding_layer_call_fn_1774460R/�,
%�"
 �
inputs���������
� "�����������
I__inference_shift_output_layer_call_and_return_conditional_losses_1774576]@A0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
.__inference_shift_output_layer_call_fn_1774585P@A0�-
&�#
!�
inputs����������
� "�����������
%__inference_signature_wrapper_1774236�45FG@A:;���
� 
���
8
numeric_input'�$
numeric_input���������
8
quality_input'�$
quality_input���������
4
shift_input%�"
shift_input���������"���
:
numeric_output(�%
numeric_output���������
:
quality_output(�%
quality_output���������
6
shift_output&�#
shift_output���������