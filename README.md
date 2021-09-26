# Recommandations vidéos

## Haddad Ayale & Hacene ISSELNANE

L’objectif est d'implémenter deux modèles de prédiction pour une problématique de recommandations de contenus vidéo. Les prédictions portent sur l'intérêt exprimé par un utilisateur sur un contenu (film, série, etc.). La recommandation consiste à proposer au client une liste de contenus ordonnés selon l'intérêt estimé.

### Description succincte de l’implémentation

Nous avons conçu le projet avec les objectifs suivants :

- Soulager la manipulation des jeux de données : Les utilisateurs peuvent charger leurs propres ensembles de données rien qu’en précisant les noms des colonnes contenant les données à utiliser.
- Optimiser les algorithmes de prédiction afin de minimiser le temp de calcul et exploiter au mieux les ressources lors de l’exécution.
- Implémenter des approches en formalisme objet.

Nous avons choisi de structurer le projet en trois fichiers :

- Un fichier « Dataset » : Celui-ci contient un ensemble de méthodes permettant de générer une structure sous forme d’une base de données contenant (user id, item id, rating associés etc…), qui par la suite sera utilisée par les algorithmes de prédiction. Permettant ainsi de construire une base commune aux deux méthodes implémentées.
- Un fichier « Baseline » : Il contient l’implémentation de la méthode décrite dans l’article [Koren].
- Un fichier « SVD++ » : Ce dernier, à l’instar du fichier « baseline » contient notre implémentation de la méthode svd++.

Nous nous sommes inspirés du célèbre package « sci-kit learn ».