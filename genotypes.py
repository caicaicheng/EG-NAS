from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    # 'none',
    'avg_pool_3x3',
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])


shapley_nas_cifar10=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
shapley_nas_imagenet = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 3), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
ci = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_3x3', 2), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 4), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))
a = Genotype(normal=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('skip_connect', 3), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
sim1 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 4), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('dil_conv_3x3', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
sim2 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0)], reduce_concat=range(2, 6))
#?
sim3 =  Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 3), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
#97.44
sim4 =  Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('max_pool_3x3', 1), ('sep_conv_3x3', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
#no good
sim5 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))
#no good
sim6 = genotype = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 3), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
#bad
sim7 =  Genotype(normal=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
sim8 =  Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 3), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

sim9 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

sim10 =  Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 3), ('max_pool_3x3', 0), ('sep_conv_3x3', 4), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

sim11= Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

sim12=  Genotype(normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))

sim13 =  Genotype(normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('max_pool_3x3', 2), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

sim14 =  Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('avg_pool_3x3', 3), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('skip_connect', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
sim15 =  Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 4), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

sim16 =  Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))
sim17 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))

sim18 =  Genotype(normal=[('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 0), ('skip_connect', 3), ('avg_pool_3x3', 4), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('avg_pool_3x3', 3), ('dil_conv_3x3', 0), ('dil_conv_3x3', 3), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
sim19 =  Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 4), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
sim20 =  Genotype(normal=[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 3), ('skip_connect', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('dil_conv_5x5', 4), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))
sim21 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

sim22=   Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

sim23 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 4), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

sim24 =  Genotype(normal=[('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 4), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
sim25 =  Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 4), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

sim26 =   Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 4), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('max_pool_3x3', 1), ('sep_conv_3x3', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

sim27 =  Genotype(normal=[('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 4), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))

sim28 =  Genotype(normal=[('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 4), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
sim29 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 4), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
sim30 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 4), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

shapley_nas = sim30



