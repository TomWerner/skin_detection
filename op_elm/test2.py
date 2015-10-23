import numpy as np
from hpelm.elm import ELM

N = 100 # number of data points
t = np.linspace(0, 4*np.pi, N)
data = np.array([[-0.03968994],
       [ 2.06378439],
       [-0.27549775],
       [ 2.11201558],
       [ 1.75628863],
       [ 2.2072026 ],
       [ 1.48230399],
       [ 3.95110109],
       [ 3.7989811 ],
       [ 3.94386006],
       [ 4.07392312],
       [ 4.45899857],
       [ 1.83719133],
       [ 3.5619939 ],
       [ 3.96341797],
       [ 3.08317235],
       [ 3.81858825],
       [ 4.07809138],
       [ 4.05971236],
       [ 2.0304482 ],
       [ 2.69969012],
       [ 1.50477087],
       [ 1.5574441 ],
       [ 1.03534912],
       [ 0.47540161],
       [ 0.04954466],
       [-1.14034123],
       [-1.76507711],
       [ 0.56558639],
       [-1.4740445 ],
       [-2.40669503],
       [-2.65625739],
       [-2.24417395],
       [-3.08574787],
       [-1.33233302],
       [-2.60943419],
       [-1.68578757],
       [-4.5171463 ],
       [-2.25260476],
       [-1.72125337],
       [-1.91562663],
       [-2.10004648],
       [-3.07726713],
       [-2.09509143],
       [-1.63337804],
       [-2.73954018],
       [-0.38757181],
       [-0.85220511],
       [ 0.26240672],
       [ 0.13672575],
       [ 1.85431992],
       [ 0.16993657],
       [ 3.23834455],
       [ 0.93764311],
       [ 1.1339608 ],
       [ 3.84843175],
       [ 2.72428074],
       [ 2.21476233],
       [ 3.67911371],
       [ 3.20556562],
       [ 2.8766396 ],
       [ 3.72116199],
       [ 1.85254957],
       [ 2.83262162],
       [ 3.2369578 ],
       [ 3.48026839],
       [ 4.04083119],
       [ 4.23330599],
       [ 3.92710604],
       [ 1.31792655],
       [ 1.81549525],
       [ 0.77802022],
       [ 1.55034898],
       [ 1.66908206],
       [ 1.24321626],
       [ 1.02011138],
       [ 0.3836505 ],
       [-0.54730131],
       [-1.75607886],
       [-1.87411834],
       [-2.18509166],
       [-2.18421888],
       [-1.86936991],
       [-1.93368369],
       [-1.08042839],
       [-0.39486789],
       [-2.06945869],
       [-1.8752493 ],
       [-3.24455534],
       [-1.46133208],
       [-2.0163139 ],
       [-1.35594818],
       [-1.59767972],
       [-0.877451  ],
       [-0.50247564],
       [ 0.0540373 ],
       [-0.52335225],
       [-0.00545229],
       [ 0.19153062],
       [ 2.59316569]])
outputs = np.array([[ 0.503     ],
       [ 0.88275303],
       [ 1.25634743],
       [ 1.61777191],
       [ 1.96121103],
       [ 2.28113871],
       [ 2.5724072 ],
       [ 2.83032989],
       [ 3.05075669],
       [ 3.23014087],
       [ 3.36559605],
       [ 3.45494273],
       [ 3.49674326],
       [ 3.49032508],
       [ 3.43579145],
       [ 3.33401983],
       [ 3.18664778],
       [ 2.99604656],
       [ 2.76528301],
       [ 2.49807021],
       [ 2.19870771],
       [ 1.87201237],
       [ 1.52324084],
       [ 1.15800498],
       [ 0.78218157],
       [ 0.40181776],
       [ 0.02303373],
       [-0.34807573],
       [-0.70553933],
       [-1.04360535],
       [-1.35683419],
       [-1.64018587],
       [-1.88910116],
       [-2.09957491],
       [-2.26822053],
       [-2.39232445],
       [-2.46988979],
       [-2.49966848],
       [-2.48118139],
       [-2.41472597],
       [-2.30137152],
       [-2.14294196],
       [-1.94198646],
       [-1.7017385 ],
       [-1.42606373],
       [-1.11939788],
       [-0.78667531],
       [-0.43324966],
       [-0.06480766],
       [ 0.3127223 ],
       [ 0.69326562],
       [ 1.07069923],
       [ 1.43895008],
       [ 1.79209287],
       [ 2.1244454 ],
       [ 2.43066   ],
       [ 2.70580955],
       [ 2.9454668 ],
       [ 3.14577558],
       [ 3.30351284],
       [ 3.41614052],
       [ 3.48184642],
       [ 3.49957329],
       [ 3.4690359 ],
       [ 3.39072561],
       [ 3.26590246],
       [ 3.09657491],
       [ 2.8854675 ],
       [ 2.63597702],
       [ 2.35211787],
       [ 2.03845745],
       [ 1.70004268],
       [ 1.34231877],
       [ 0.97104163],
       [ 0.59218526],
       [ 0.2118456 ],
       [-0.16385756],
       [-0.52887899],
       [-0.87734538],
       [-1.20364977],
       [-1.50254179],
       [-1.76921217],
       [-1.99937007],
       [-2.18931218],
       [-2.33598224],
       [-2.43702029],
       [-2.49080058],
       [-2.49645777],
       [-2.45390084],
       [-2.36381453],
       [-2.22764838],
       [-2.04759334],
       [-1.82654657],
       [-1.5680648 ],
       [-1.27630709],
       [-0.95596793],
       [-0.61220171],
       [-0.25053974],
       [ 0.12319869],
       [ 0.503     ]])

np.random.seed(1)
elm = ELM(data.shape[1], outputs.shape[1])
elm.add_neurons(30, "lin")
elm.add_neurons(30, "sigm")
elm.add_neurons(30, "tanh")
print(elm.neurons)
elm.train(data, outputs)
Y = elm.predict(data)
print(sum(np.square(Y - outputs)))
# print(elm.Beta)