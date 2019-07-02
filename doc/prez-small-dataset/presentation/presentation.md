---
title: Apprentissage sur un petit ensemble de données
#subtitle:
author: Bousquet Jérémie, Hmamouche Youssef
#date: LPL, Aix En Provence, 25.04.2019

output:
  beamer_presentation:
    theme: "AnnArbor"
    colortheme: "dolphin"
    fonttheme: "structurebold"

header-includes:
    - \usepackage{hyperref}
    - \usepackage{multirow}
    - \usepackage{subfig}
    - \usepackage{pgfkeys}
    - \usepackage{amsmath}
---

## Apprentissage automatique

Algorithmes permettant de résoudre un problème, une tâche, en apprenant des données - lorsqu'il n'est pas possible
ou complexe d'écrire un algorithme pour résoudre la tâche.

$$
X = \begin{pmatrix} 
x_{11} & x_{12} & ... & x_{1p} \\
x_{21} & x_{22} & ... & x_{2p} \\
... & ... & ... & ... \\
x_{n1} & x_{n2} & ... & x_{np} 
\end{pmatrix}
\text{ et } 
y =  \begin{pmatrix} 
y_{1} \\
y_{2}  \\
...  \\
y_{n}  
\end{pmatrix} $$ 

* Chaque ligne de $X$ est une donnée structurée (un exemple), composée de $p$ variables (features)
* A chaque exemple est associée une cible $y_{i}$, ce que l'on souhaite prédire
* $f(X)=y+\epsilon$ est ce que l'algorithme cherche à approximer au mieux ($\epsilon$ l'erreur)
* Si les $y_{i}$ sont continus on parle de tâche de régression, s'ils sont discrets on parle de tâche de 
classification

## Apprentissage automatique

Classification: reconnaître un objet dans une image (chaque ligne de X est une image, les $p$
variables représentant les pixels de l'image, et par exemple
$y \in \{chien, chat, ours, ...\}$)

Régression: prédiction du prix de vente d'une maison en fonction de critères (les $p$ variables sont les critères, superficie, nombre de chambres, etc, et
$y$ est le prix de vente)

Les programmes utilisant ces algorithmes ont deux phases:

* Apprentissage: on approxime $f(X)=y$, $X$ et $y$ étant connus (apprentissage supervisé)

* Inférence: on prédit $y$ à partir de $f$ apprise et de nouvelles données $X^{'}$ pour lesquelles $y^{'}$ 
peut être connu (pour tester l'apprentissage) ou pas (problème réel).

D'autres formes d'apprentissage existent, par ex. non supervisé (pas de $y$, par exemple le clustering
qui vise à détecter des groupements au sein des $X$)

## Apprentissage automatique

* Biais inductif: on commence par faire une hypothèse dans le choix d'une famille de fonctions $\mathbb{H}$
 pour $f$ (ex. pour la
régression: polynomiale, logistique..., pour la classification: Support Vector Machines, Bayésien, 
réseau de neurones ...)

* L'erreur commise sur plusieurs jeux de données peut se décomposer en biais et variance. 

* Le biais est l'écart entre la fonction de prédiction moyenne apprise sur plusieurs jeux de données, 
et la fonction qui minimise l'erreur d'apprentissage (perte)

* La variance est l'écart entre la fonction de prédiction moyenne apprise sur plusieurs jeux de données,
et une fonction de prédiction apprise sur un jeu de données

## Apprentissage automatique

Le biais et la variance sont liés:

* Si la complexité du modèle $\mathbb{H}$ choisi est faible, le biais sera important et la variance faible

* Si la complexité du modèle $\mathbb{H}$ choisi est forte, le biais sera faible mais la variance importante

\begin{figure}[H]
\includegraphics[width=0.75\textwidth]{figs/biais-variance.png}
\end{figure}

Si le biais est trop important on parle de sous-apprentissage. Si la variance est trop importante on parle de 
sur-apprentissage (overfitting).


## Overfitting

Le terme "overfitting" correspond au cas où le prédicteur modélise trop étroitement les données d'apprentissage (jusqu'à
les apprendre "par coeur"), et de fait ne généralise pas ou très mal ses capacités sur de nouvelles données.

* comment mesurer l'overfitting

   * Evaluer la différence entre les erreurs des modèles sur les données de validation (entrainement) et les données de test.
    Example, utiliser des tests de signification statistiques.

   * Employer des méthodologies non biaisées (qui révèlent overfitting ou underfitting, dans l'apprentissage et la sélection
   du modèle, voir par ex. <a href="http://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf">Cawley et al. (2017)</a>)

   * Ne pas se limiter à la mesure de la précision


## Overfitting

* comment éviter l'overfitting

  * poser des contraintes sur les hyper-paramètres (lorsque cela est pertinent/possible). Pour certains classifieurs des paramètres 'extrêmes' (grands/petits) peuvent encourager l'overfitting

  * ajouter une régularisation (contrainte pour limiter la complexité du modèle)

  * augmenter les données


## Application - Prédiction de l'activité cérébrale en fonction des signaux multimodaux
* Méthode utilisée : k-fold cross-validation avec ensemble de test et de validation.
* Problèmatique : la cross validation pose quelques problèmes pour les données séquentielles car elle ne tient pas en compte l'ordre chronologiques des données,
mais on peut la faire marcher dans notre cas si on considère chaque  conversation comme un sous-ensemble sous l'hypothèse que l'ordre des conversations n'est pas important.

## Application - Prédiction de l'activité cérébrale en fonction des signaux multimodaux
\small
* Première stratégie :  construire un seul modèle pour toutes les conversations.

* Dans ce cas, on peut découper les données en 4 blocks (comme découpé lors de l'expérience d'IRMf), chaque block contient 6 conversations. Sur chaque block on peut appliquer une k-fold cross-validation en gardant une seule conversation comme données de test. Pour les autres, on change aléatoirement l'ordre des conversations à chaque fois et fixant une conversation pour la validation, et on répète ce processus, jusqu'à ce que chaque conversation des données d'entrainement est utilisée une fois pour la validation.
* Deuxième stratégie :  deux modèles séparés, un pour les conversations humain-humain (HH) et l'autre pour les conversations humain-robot (HR).

## Application - Prédiction de l'activité cérébrale en fonction des signaux multimodaux
### Modélisation 1

* 20 conversations pour l'entrainement, et 4 conversations de test (2 HH et 2 HR).
* 5-fold cross-validation avec ensemble de test et de validation sur les 4 blocks :
   * Division des conversations en 4 blocks.
   * Sur chaque block, une conversation est extraite comme données de test.
   * Application d'une 5-fold cross-validation sur le reste des conversations.

## Application - Prédiction de l'activité cérébrale en fonction des signaux multimodaux
### Modélisation 1 : un modèle pour toutes les conversations
\begin{center}
\includegraphics[width=0.95\textwidth]{figs/cross.pdf}
\end{center}

## Application - Prédiction de l'activité cérébrale en fonction des signaux multimodaux
### Modélisation 2 : deux modèles selon le type des conversations
\begin{figure}[H]
\includegraphics[width=0.95\textwidth]{figs/cross2.pdf}
\end{figure}

## Application - Prédiction du sentiment de (co)présence

Procédure:

* discrétisation des scores de présence/co-présence en problèmes de classification binaire

* 100x 90% train / 10% test splits aléatoires stratifiés (= conservant les proportions de chaque classe)

  * 10-fold cross-validation sur l'ensemble train pour la recherche d'hyper-paramètres du modèle

  * évaluation de la capacité de prédiction sur l'ensemble test (non vu lors de l'apprentissage)

  * moyennage des scores de test sur les 100 splits (avec calcul de l'erreur standard sur la moyenne)

## Application - Prédiction du sentiment de (co)présence

_Overfitting: cas du Support Vector Machines (SVM) (1/2)_

Le classifieur SVM tente de séparer les données par un hyper-plan, avec la contrainte d'avoir une marge minimale
(entre l'hyper-plan et les données) qui soit maximale. Le paramètre C détermine un compromis entre maximiser cette marge,
et autoriser la mauvaise classification de certains points : plus C est grand, plus les points mal classés sont exclus,
et plus la marge aura tendance à être petite.

\begin{figure}[H]
\includegraphics[width=0.75\textwidth]{figs/GbW5S.png}
\end{figure}

## Application - Prédiction du sentiment de (co)présence

_Overfitting: cas du Support Vector Machines (SVM) (2/2)_

Pour une valeur de C grande, SVM tentera de classer correctement chaque exemple d'apprentissage, au prix de la marge et
possiblement d'une meilleure capacité de généralisation:

\begin{figure}[H]
\includegraphics[width=0.95\textwidth]{figs/svm-overfit.png}
\end{figure}

## Méthodes et techniques adaptées aux petits ensembles de données

* Utiliser des algorithmes empiriquement plus adaptés aux petits ensembles de données (non exhaustif: Random Forests,
Naïve Bayes ...)

* Augmenter les données, pour:

  * obtenir un dataset plus grand

  * et/ou réduire les déséquilibres entre classes

* Diminuer l'influence du bruit / le biais, retirer les outliers

  * auditer les exemples d'apprentissage existants

  * régulariser l'apprentissage

 Sur un faible volume de données le bruit, les outliers, peuvent avoir un impact important.


## Techniques pour la génération de nouvelles données

* Random sampling

Re-sampling (tirage avec remise), utilisé notamment pour résoudre les déséquilibres de classe en répliquant
des échantillons de la classe minoritaire.

* SMOTE, ADASYN

Synthèse de nouveaux examples par interpolation d'exemples existants.

* Réseaux de neurones dont les réseaux antagonistes génératifs (GAN)

Un réseau apprend à générer des données, et est corrigé par un réseau apprenant à discriminer vraie donnée et donnée
générée (apprentissage souvent difficile).


## Application - Prédiction de l'activité cérébrale en fonction des signaux multimodaux
### Génération de nouvelles données
* Il serait intéréssant d'ajouter de nouvelles données si cela permet d'améliorer la qualité des prédictions.
* Pour le moment, pour chaque sujet, nous avons 1200 observations, où chaque conversation contient 50 observations.
* Ces observations présentent des auto-corrélations pour la plupart des variables.
* Par conséquent, il faut générer des données de manière à tenir en compte ces auto-corrélations.



## Application - Prédiction du sentiment de (co)présence

* GAN

  * peut nécessiter de grands volumes de données

    * approche "fine-tuning", mais possible seulement si le domaine/la tâche est semblable, compliqué ici

  * qualité / pertinence des données générées difficile à évaluer

## Application - Prédiction du sentiment de (co)présence

* Random Sampling, SMOTE, ADASYN, ...

  * Faciles à mettre en oeuvre, variables continues interpolables

  * Variables catégorielles: méthodes pour déterminer la catégorie de la nouvelle donnée (plus proches voisins ...)

  * expérimentalement pas de réel avantage mesuré du moment que les métriques prennent en compte le déséquilibre de classes:

\begin{figure}[H]
\includegraphics[width=0.65\textwidth]{figs/Oversampling-method_Presence_NB-G_test.png}
\end{figure}
