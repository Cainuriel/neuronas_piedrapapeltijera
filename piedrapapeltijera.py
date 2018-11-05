from random import choice
from sklearn.neural_network import MLPClassifier 

options = ["Piedra","Tijera","Papel"]
result = 0
# array para realizar los test
test = [ ["Piedra","Piedra",0],
		["Piedra","Tijera",1],
		["Tijera","Papel",1], 
		["Piedra","Papel",2],
		["Tijera","Piedra",2],
		["Papel","Piedra",1],
		["Piedra","Papel",2]]


def whoisthewiner(jug1, jug2):
	if jug2 == jug1:
		result = 0
	elif jug1 == "Piedra" and jug2 == "Tijera":
		result = 1
	elif jug1 == "Piedra" and jug2 == "Papel":
		result = 2
	elif jug1 == "Papel" and jug2 == "Piedra":
		result = 1
	elif jug1 == "Papel" and jug2 == "Tijera":
		result= 2
	elif jug1 == "Tijera" and jug2 == "Piedra":
		result = 2
	elif jug1 == "Tijera" and jug2 == "Papel":
		result = 1

	return result

"""for play in test:
	print("Jugador 1: %s Jugador 2: %s Ganador: %s Validacion: %s" %(
		play[0], play[1], whoisthewiner(play[0],play[1]),play[2])) """

def get_choice():
	return choice(options)

for play in range(10):
	player_1 = get_choice()
	player_2 = get_choice()
	print("Jugador 1: %s Jugador 2: %s Ganador: %s" %(
		player_1, player_2, whoisthewiner(player_1,player_2)
		))


# transformamos las opciones en binario.
def str_to_list(option):
	if option == "Tijera":
		res = [1,0,0]
	elif option == "Piedra":
		res = [0,1,0]
	else:
		res = [0,0,1]

	return res 

datax = list(map(str_to_list, ["Tijera","Piedra","Papel"]))
datay = list(map(str_to_list,["Papel","Tijera","Piedra"]))

clf = MLPClassifier(verbose = False, warm_start = True)

# metodo que entrena a la red neuronal. 
model = clf.fit([datax[0]],[datay[0]])

print(model)

# funcion para que aprenda segun vaya jugando. 
# parametros: iters: iterador para generar diez partidas. 
def play_and_learn(iters = 10, debug = False):
	score = {"win": 0, "loose": 0}
	datax = []
	datay = []

	for i in range(iters):
		# elegimos una opcion para el jugador 1.
		player_1 = get_choice()
		# y segun haya elegido, la maquina tendra que aprender a adivinar
		# cual es la probabilidad de opcion de respuesta, ante la opcion del player 1.
		# colocamos el cero para quedarnos con el primer resultado. 
		predict = model.predict_proba([str_to_list(player_1)])[0]

		# condicionamos los porcentajes. 
		if predict[0] >= 0.95:
			player_2 = options[0]
		elif predict[1] >= 0.95:
			player_2 = options[1]
		elif predict[2] >= 0.95:
			player_2 = options[2]
		else:
			player_2 = get_choice()

		# pintamos la opcion del primer jugador. preguntamos al modelo y pintamos la salida
		# de las probabilidades, tantos por ciento; y por ultimo, pintamos la opcion elegida
		if debug == True:
			print("Jugador 1: %s Jugador 2 (Modelo): %s --> %s " % (player_1,predict,player_2))

		iswinner = whoisthewiner(player_1,player_2)

		if debug == True:
			print("0: empate. 1: Gana Jugador 1 2: Gana jugador 2. Resultado: %s" % iswinner)

		# Si la maquina gana tendremos que hacer que recuerde. Recordamos que prentedemos que
		# la maquina aprenda a traves de sus victorias jugando. por tanto:
		# la opcion es el jugador segundo, que es la maquina.
		if iswinner == 2:
			# Guardamos la tirada del jugador 1 y la tirada ganadora de la maquina. 
			datax.append(str_to_list(player_1))
			datay.append(str_to_list(player_2))

			# apuntamos la victoria. 
			score["win"] += 1
		else:
			score["loose"] += 1

	return score, datax, datay

# probamos la funcion play_and_learn. Ponemos el debug en True para ver las iteracciones.

score, datax, datay = play_and_learn(1, debug = True)
print("Jugador 1 ",datax)
print("Respuesta ",datay)

# la ecuacion calcula las victorias en relacion del total de perdidas y ganadas. 
print("Resultado: %s %s %%" % (score,(score["win"]*100/(score["win"]+score["loose"]))))

# en el caso de tener algo aprendido re-entrenamos 
if len(datax):
	model =  model.partial_fit(datax,datay)

i = 0

# creamos un historico de los porcentajes para comprobar como va aprendiendo.
historico = []

while True:
	i+= 1
	# generamos mil iteraciones y el debug en falso porque ya no queremos hacer comprobaciones. 
	score, datax, datay = play_and_learn(1000, debug = False)
	porcentaje = (score["win"]*100/(score["win"]+score["loose"]))
	# metemos los porcentajes para comprobar como va mejorando
	historico.append(porcentaje)
	# ahora imprimimos la iteraccion, el resultado y el porcentaje
	print("Iteraccion: %s - Resultado: %s - Porcentaje: %s %%" %(i, score, porcentaje))

	# si hay jugadas hay que re-entrenar. 
	if len(datax):
		model =  model.partial_fit(datax,datay)

# y para parar el while miraremos las ultimas nueve partidas en el historico
# si estas son del cien por cien, entonces paramos. 
	if sum(historico[-9:])==900:
		break

  


