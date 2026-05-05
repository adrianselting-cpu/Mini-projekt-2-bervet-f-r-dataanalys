import numpy as np
import matplotlib.pyplot as plt

#A: traindigits innehåller bilderna, där varje kolumn är en bild på 784 (28x28) pixlar. De 240 000 kolumnerna är 400 olika exempel på hur folk har ritat den siffran.
#trainlabels innehåller svaret, dvs vad varje bild faktiskt föreställer. Alltså facit

TrainDigits = np.load('HandwrittenDigits/TrainDigits.npy')
TrainLabels = np.load('HandwrittenDigits/TrainLabels.npy')
TestDigits = np.load('HandwrittenDigits/TestDigits.npy')
TestLabels = np.load('HandwrittenDigits/TestLabels.npy')

print("TrainDigits shape:", TrainDigits.shape)  # Ska vara (784, 240000)
print("TrainLabels shape:", TrainLabels.shape)  # Ska vara (240000,)
print("TestDigits shape:", TestDigits.shape)    # Ska vara (784, 40000)
print("TestLabels shape:", TestLabels.shape)    # Ska vara (40000,)

TrainLabels = TrainLabels[0]
TestLabels = TestLabels[0]

#grundiden är att 
#1. Vi har A (träningsdata vi redan vet svaret på)
#2. Vi kör SVD på A
#3. Vi får ut U och Σ
#4. Vi sparar Uk (de första 15 kolumnerna av U)

##############################
#Del 1, dela upp träningsdatan per siffra
###############################
TrainPerDigit = [] #Skapar en tom lista som kommer hålla 10 matriser (en per siffra)
for i in range(10): 
    index = (TrainLabels == i) #jämför varje element i TrainLabels med värdet i och returnerar True/False-array(2400 element)
    digits = TrainDigits[:, index] #Plockar ut alla kolumner från TrainDigits där index är True. Alltså alla bilder av siffran i, ca 24000 bilder 
    A = digits[:, :400] #Vi tar bara 400 första bilderna. (784x400)
    TrainPerDigit.append(A)

#Del 2, träning, beräkna SVD för varje siffra och plocka ut för varje siffra Uk (784x15)som beskriver de 15 viktigaste mönstren för siffran (kolumner med störst singulärvärde).

kmax = 15
Uk_list = []
for i in range(10):
    A = TrainPerDigit[i]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Uk = U[:, :kmax] 
    Uk_list.append(Uk)
    print(f"SVD klar för siffra {i}, Uk shape: {Uk.shape}")



####################################
#Task 2
####################################

# Vi plottar singulärvärdena för siffrorna 3 och 8
for i in [3, 8]:
    A = TrainPerDigit[i]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    plt.figure(figsize=(8, 4))
    plt.plot(S, 'o-', markersize=3)
    plt.title(f"Singulärvärden för siffra {i}")
    plt.xlabel("Index")
    plt.ylabel("Värde")
    plt.grid(True)
    plt.show()

#Vi plottar bilderna för siffrorna 3 och 8 för de tre första singulära bilderna
for i in [3, 8]:
    Uk = Uk_list[i] 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"De första tre singulära bilderna för siffra {i}")
    
    for k in range(3):
        # Ta kolumn k, reshape till 28x28 = 784
        img = Uk[:, k].reshape(28, 28)
        axes[k].imshow(img, cmap='gray')
        axes[k].set_title(f"u_{k+1}")
        axes[k].axis('off')
    plt.show()


# Skapa en lista för att spara resultaten
# Rader = siffror (0-9), Kolumner = k-värden (5-15)
accuracy_matrix = np.zeros((10, 11)) 

# Loopa igenom alla k-värden som efterfrågas
for k in range(5, 16):
    correct_predictions = np.zeros(10)
    total_images_per_digit = np.zeros(10)
    
    # För att det ska gå snabbt, använd tips 4 & 5 i Miniproject 2-2.pdf
    # Istället för att loopa varje bild, kan man räkna ut residualer för hela matriser
    for i in range(10):
        # Ta ut de k första kolumnerna för siffra i
        Uk = Uk_list[i][:, :k]
        
        # Beräkna projektionsfelet för ALLA 40 000 testbilder samtidigt
        # residual = || d - Uk * (Uk.T * d) ||
        projection = Uk @ (Uk.T @ TestDigits)
        residuals = np.linalg.norm(TestDigits - projection, axis=0)
        
        # Spara dessa residualer i en matris (10 rader x 40000 kolumner)
        if i == 0:
            all_residuals = residuals
        else:
            all_residuals = np.vstack([all_residuals, residuals])
            
    # Hitta vilken siffra (index 0-9) som hade lägst residual för varje bild
    predictions = np.argmin(all_residuals, axis=0)
    
    # Jämför med facit (TestLabels)
    for digit in range(10):
        mask = (TestLabels == digit)
        correct = np.sum(predictions[mask] == digit)
        total = np.sum(mask)
        accuracy_matrix[digit, k-5] = (correct / total) * 100

    print(f"Klar med beräkning för k = {k}")


#presentera res
print("\nSuccess rate per digit (%) för k=15:")
for i in range(10):
    # Vi använder accuracy_matrix och kollar på sista kolumnen (-1) som är k=15
    print(f"Siffra {i}: {accuracy_matrix[i, -1]:.2f}%")

# 2. För att svara helt på Task 3 bör du visa hela tabellen
print("\nFullständig tabell (Rader=Siffra, Kolumner=k från 5 till 15):")
print(accuracy_matrix)