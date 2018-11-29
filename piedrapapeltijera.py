from random import choice
from sklearn.neural_network import MLPClassifier 
import pickle
options = ["piedra", "tijeras", "papel"]
result = 0


def search_winner(p1, p2):
    if p1 == p2:
        result = 0
    
    elif p1 == "piedra" and p2 == "tijeras":
        result = 1
    elif p1 == "piedra" and p2 == "papel":
        result = 2
    elif p1 == "tijeras" and p2 == "piedra":
        result = 2
    elif p1 == "tijeras" and p2 == "papel":
        result = 1
    elif p1 == "papel" and p2 == "piedra":
        result = 1
    elif p1 == "papel" and p2 == "tijeras":
        result = 2
        
    return result

print("probando funcion de ganador",search_winner("papel", "tijeras"))

testear = [
    ["piedra", "piedra", 0],
    ["piedra", "tijeras", 1],
    ["piedra", "papel", 2]
]
   
    

for partida in testear:
    print("player1: %s player2: %s Winner: %s Validation: %s" % (
        partida[0], partida[1], search_winner(partida[0], partida[1]), partida[2]
    ))

def get_choice():
    return choice(options)

for i in range(10):
    player1 = get_choice()
    player2 = get_choice()
    print("player 1: %s player 2: %s Winner is: %s " % (
        player1, player2, search_winner(player1, player2)
    ))

# pasamos los elementos de las opciones a binario. 
def str_to_list(option):
    if option=="piedra":
        res = [1,0,0]
    elif option=="tijeras":
        res = [0,1,0]
    else:
        res = [0,0,1]
    return res

data_X = list(map(str_to_list, ["piedra", "tijeras", "papel"]))
data_y = list(map(str_to_list, ["papel", "piedra", "tijeras"]))

print(data_X)
print(data_y)

clf = MLPClassifier(verbose=False, warm_start=True)


model = clf.fit([data_X[0]], [data_y[0]])
print("nuestro modelo: ",model)

def play_and_learn(iters=10, debug=False):
    score = {"win": 0, "loose": 0}
    
    data_X = []
    data_y = []
    
    for i in range(iters):
        player1 = get_choice()
        
        predict = model.predict_proba([str_to_list(player1)])[0]
        
        if predict[0] >= 0.95:
            player2 = options[0]
        elif predict[1] >= 0.95:
            player2 = options[1]
        elif predict[2] >= 0.95:
            player2 = options[2]
        else:
            player2 = get_choice()
            
        if debug==True:
            print("Player1: %s Player2 (modelo): %s --> %s" % (player1, predict, player2))
        
        winner = search_winner(player1, player2)
        if debug==True:
            print("Comprobamos: p1 VS p2: %s" % winner)
        
        if winner==2:
            data_X.append(str_to_list(player1))
            data_y.append(str_to_list(player2))
            
            score["win"]+=1
        else:
            score["loose"]+=1
        
    return score, data_X, data_y

score, data_X, data_y = play_and_learn(1, debug=True)
print(data_X)
print(data_y)
print("Score: %s %s %%" % (score, (score["win"]*100/(score["win"]+score["loose"]))))
if len(data_X):
    model = model.partial_fit(data_X, data_y)

i = 0
historic_pct = []
while True:
    i+=1
    score, data_X, data_y = play_and_learn(1000, debug=False)
    pct = (score["win"]*100/(score["win"]+score["loose"]))
    historic_pct.append(pct)
    print("Iter: %s - score: %s %s %%" % (i, score, pct))
    
    if len(data_X):
        model = model.partial_fit(data_X, data_y)
    
    if sum(historic_pct[-9:])==900:
        break


print(model.predict_proba([str_to_list("piedra")]))

# guarda el modelo. 
filename = 'piedrapapeltijera_model.sav'
pickle.dump(model, open(filename, 'wb'))

