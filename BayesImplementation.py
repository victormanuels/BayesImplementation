#!/usr/bin/env python
# coding: utf-8

# # StopWords
# Se crea un arreglo con las stopwords que seran elimindas

# In[ ]:


stopWords=['i',
'me',
'my',
'myself',
'we',
'our',
'ours',
'ourselves',
'you',
'your',
'yours',
'yourself',
'yourselves',
'he',
'him',
'his',
'himself',
'she',
'her',
'hers',
'herself',
'it',
'its',
'itself',
'they',
'them',
'their',
'theirs',
'themselves',
'what',
'which',
'who',
'whom',
'this',
'that',
'these',
'those',
'am',
'is',
'are',
'was',
'were',
'be',
'been',
'being',
'have',
'has',
'had',
'having',
'do',
'does',
'did',
'doing',
'a',
'an',
'the',
'and',
'but',
'if',
'or',
'because',
'as',
'until',
'while',
'of',
'at',
'by',
'for',
'with',
'about',
'against',
'between',
'into',
'through',
'during',
'before',
'after',
'above',
'below',
'to',
'from',
'up',
'down',
'in',
'out',
'on',
'off',
'over',
'under',
'again',
'further',
'then',
'once',
'here',
'there',
'when',
'where',
'why',
'how',
'all',
'any',
'both',
'each',
'few',
'more',
'most',
'other',
'some',
'such',
'no',
'nor',
'not',
'only',
'own',
'same',
'so',
'than',
'too',
'very',
'can',
'will',
'just',
'don',
'should',
'now','subject','-','--','\n'];


# Se crea un método el cuál eliminara StopWords y caracteres especiales

# In[ ]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
def removeStopWords(word):
    #return word.lower()
    word=word.lower()
    word=re.sub('[^A-Za-z0-9- ]+', ' ',word)
    stop_words = stopWords
    
    
    
    wordTokens = word_tokenize(word)
    wordFiltered = [w for w in wordTokens if not w in stop_words]
    return  ' '.join(wordFiltered) 


# # Lectura de dataset
# 
# Se crea arreglo con la informacion del dataset, la informacion debe estar almacenada dentro de carpetas y en cada carpeta dos subcarpetas, una para smap y otra para ham

# In[ ]:


import pandas as pd
import os
import re
import codecs

columns = ['email', 'class']
dataInfo = []

directory="/entornDataset"
i=0;
for currentFile in os.listdir(directory):
    fullDirectory=directory+"/"+currentFile
    if (os.path.isdir(fullDirectory)):
        fullDirectorySpam=fullDirectory+"/spam"
        for spamFile in os.listdir(fullDirectorySpam):
            sf=open(fullDirectorySpam+"/"+spamFile, "r",encoding='utf-8', errors='ignore')
            if sf.mode == 'r':
                contents =sf.read()
                dataInfo.append([removeStopWords(contents),"spam"])
        fullDirectoryHam=fullDirectory+"/ham"
        
        for hamFile in os.listdir(fullDirectoryHam):
            #print(fullDirectoryHam+"/"+hamFile)
            hf=open(fullDirectoryHam+"/"+hamFile, "r",encoding='utf-8', errors='ignore')
            if hf.mode == 'r':
                contents =hf.read()
                dataInfo.append([removeStopWords(contents),"ham"])


# # Procesado de  la información
# Se filtran los correos etiquetados como spam de los etiquetados como ham, 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
fullData = pd.DataFrame(dataInfo, columns=columns)
spamData=[]
hamData=[]

spamData = [row['email'] for index,row in fullData.iterrows() if row['class'] == 'spam']
hamData = [row['email'] for index,row in fullData.iterrows() if row['class'] == 'ham']

spamDataSize=len(spamData)
hamDataSize=len(hamData)
totalDataLen=hamDataSize+spamDataSize
 
spamDataPr=spamDataSize/totalDataLen
hamDataPr=hamDataSize/totalDataLen


# Tanto como Ham y Spam se conveiren en bolsas de palabras

# In[ ]:


bagSpam = CountVectorizer()
bagSpams2 = bagSpam.fit_transform(spamData)
spamCount = pd.DataFrame(bagSpams2.toarray(), columns=bagSpam.get_feature_names())

bagham = CountVectorizer()
baghams2 = bagham.fit_transform(hamData)
hamCount = pd.DataFrame(baghams2.toarray(), columns=bagham.get_feature_names())



# # Contar repeticiones de palabras
# 
# Se cuentan todas las repeticienes de las palabras en ham y spam

# In[ ]:



hamTotal=hamCount.sum(axis=None, skipna=None, level=None, numeric_only=None, min_count=0)
spamTotal=spamCount.sum(axis=None, skipna=None, level=None, numeric_only=None, min_count=0)
sumAllHam=0
sumAllSpam=0
for x in range(len(hamTotal)):
    sumAllHam+=hamTotal[x]
    
for x in range(len(spamTotal)):
    sumAllSpam+=spamTotal[x]


probabilitiesTotalHam=[]
probabilitiesTotalSpam=[]
bagHamSpamWords=[]
for i in range(len(hamTotal)):
    probabilitiesTotalHam.append([hamCount.columns[i],(hamTotal[i]/len(hamTotal))])
    bagHamSpamWords.append(hamCount.columns[i])
for i in range(len(spamTotal)):
    probabilitiesTotalSpam.append([spamCount.columns[i],(spamTotal[i]/len(spamTotal))])
    bagHamSpamWords.append(spamCount.columns[i])


# In[ ]:


bagHamSpamWords = set(bagHamSpamWords)


# # Calcular la probabilidad
# Se crea un metodo el cual recibe un vector de palabras la cual sera clasificada
# 

# In[ ]:


def computedProbability(dataToClasify,isSpam):
    proabilitysWords=[]
    if isSpam:
        for x in range(len(dataToClasify.columns)):
            existWord=False
            for y in range(len(probabilitiesTotalSpam)):
                #print(dataToClasify.columns[x])
                if dataToClasify.columns[x]==probabilitiesTotalSpam[y][0]:
                    #print(dataToClasify.columns[x])
                    existWord=True
                    #print(probabilitiesTotalSpam[y])
                    proabilitysWords.append(probabilitiesTotalSpam[y][1]**hamToClassifyHamCount.iloc[0][x])
            if existWord==False:
                #print("existWordF")
                break
    else:
        for x in range(len(dataToClasify.columns)):
            existWord=False
            for y in range(len(probabilitiesTotalHam)):
                if dataToClasify.columns[x]==probabilitiesTotalHam[y][0]:
                    existWord=True
                    proabilitysWords.append(probabilitiesTotalHam[y][1]**hamToClassifyHamCount.iloc[0][x])
            if existWord==False:
                break
    return [existWord,proabilitysWords]
    

            
        
    


# # Laplace Smoothing
# En ocasiones, en casos de que una palabra no tenga coincidencias, se debe de hacer LaplaceSmoothing para obtener las probabilidades, de lo contario el resultado sera 0

# In[ ]:


def computedProbabilityLaplaceSmoothing (dataToClasify,isSpam):
    proabilitiesWords=[]
    if isSpam:
        for x in range(len(dataToClasify.columns)):
            num=0
            for y in range (len(spamTotal)):
                if dataToClasify.columns[x]==spamCount.columns[y]:
                    num=spamTotal[y]
                    break
            #print("num: "+str(num)+" len(spamToClassifySpamCount.columns): "+str(len(spamToClassifySpamCount.columns))+" len(bagHamSpamWords)"+str(len(bagHamSpamWords)) )
            prob=((num+1)/(sumAllSpam+len(bagHamSpamWords)))
            prob=prob**hamToClassifyHamCount.iloc[0][x]
            print(prob)
            proabilitiesWords.append(prob)
    else:
        for x in range(len(dataToClasify.columns)):
            num=0
            for y in range (len(hamTotal)):
                if dataToClasify.columns[x]==hamCount.columns[y]:
                    num=hamTotal[y]
                    break
            #print("num: "+str(num)+" sumAllHam: "+str(sumAllHam)+" len(bagHamSpamWords)"+str(len(bagHamSpamWords) ))
            prob=((num+1)/(sumAllHam+len(bagHamSpamWords)))
            prob=prob**hamToClassifyHamCount.iloc[0][x]
            proabilitiesWords.append(prob)
     
    return proabilitiesWords
        


# # Implementación
# 

# In[ ]:


newEmail = input("Introduce una cadena de texto: ")


# # Procesar el dato a clasificar
# Primero se remueven las stop words, posteriormente se convierte en un dataframe
# 

# In[ ]:


hamToClassify=[newEmail]
hamToClassify[0]=removeStopWords(hamToClassify[0])

spamToClassify=[newEmail]
spamToClassify[0]=removeStopWords(spamToClassify[0])


hamToClassifyHam = CountVectorizer()
hamToClassifyHam2 = hamToClassifyHam.fit_transform(hamToClassify)
hamToClassifyHamCount = pd.DataFrame(hamToClassifyHam2.toarray(), columns=hamToClassifyHam.get_feature_names())


spamToClassifySpam = CountVectorizer()
spamToClassifySpam2 = spamToClassifySpam.fit_transform(spamToClassify)
spamToClassifySpamCount = pd.DataFrame(spamToClassifySpam2.toarray(), columns=spamToClassifySpam.get_feature_names())


# # Calculan las probabilidades
# Si el método probabilitiesResultHam, regresa falso, significa que se obtuvo una probabilidad de 0, entonces se aplica el Laplace Smoothing

# In[ ]:


probabilitiesResultHam=computedProbability(hamToClassifyHamCount,False)
#print(probabilitiesResultHam[1])
priabilitiesFinalHam=[]
if probabilitiesResultHam[0]==False:
    print("computedProbabilityLaplaceSmoothing1")
    priabilitiesFinalHam=computedProbabilityLaplaceSmoothing(hamToClassifyHamCount,False)
else:
    priabilitiesFinalHam=(probabilitiesResultHam[1])
        
        
probabilitiesResultSpam=computedProbability(spamToClassifySpamCount,True)
priabilitiesFinalSpam=[]

if probabilitiesResultSpam[0]==False:
    priabilitiesFinalSpam=computedProbabilityLaplaceSmoothing(spamToClassifySpamCount,True)
else:
    priabilitiesFinalSpam=probabilitiesResultSpam[1]
    


# # Calcultar probabilidades
# Por ultimo, se multiplican las probabilidads de cada palabra por el total de la clase

# In[ ]:


hamProbability=1
for y in range ((len(priabilitiesFinalHam))):
    hamProbability+=hamProbability*priabilitiesFinalHam[y]
#print("\n")
spamProbability=1
for y in range ((len(priabilitiesFinalSpam))):
    #print(priabilitiesFinalSpam[y])
    spamProbability+=spamProbability*priabilitiesFinalSpam[y]


# In[ ]:


print(spamProbability)
print(hamProbability)


# In[ ]:


spamProbability=spamProbability*spamDataPr
hamProbability=hamProbability*hamDataPr


# In[ ]:


print("La probabilidad de spam es: "+str(spamProbability))
print("La probabilidad de ham es: "+str(hamProbability))


# In[ ]:


if spamProbability<hamProbability:
    print("Es ham")
else:
    print("Es spam")

    

