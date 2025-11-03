# Unsupervised Learning Project

![](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)

![](https://github.com/claudiaewilson/Unsupervised_Learning_Project/blob/main/anime_1.png)

## Table of contents
* [1. Project Overview](#project-description)
* [2. Dataset](#dataset)
* [3. Packages](#packages)
* [4. Environment](#environment)
* [5. Team Members](#team-members)

## 1. Project Overview <a class="anchor" id="project-description"></a>
In today’s technology-driven world, recommender systems are socially and economically critical for ensuring that individuals can make appropriate choices surrounding the content they engage with on a daily basis. One application where this is especially true surrounds movie content recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.

…ever wondered how Netflix, Amazon Prime, Showmax, Disney and the likes somehow know what to recommend to you?

Well, you are about to find out! In this project, we require you to use all your skills to build a collaborative and content-based recommender system for a collection of anime titles, capable of accurately predicting how a user will rate an anime title they have not yet viewed, based on their historical preferences.

The dataset consists of thousands of users and thousands of anime titles, gathered from myanimelist.net.

![](https://github.com/claudiaewilson/Unsupervised_Learning_Project/blob/main/anime_2.png)

## 2. Dataset <a class="anchor" id="dataset"></a>
This dataset contains information on anime content (movies, television series, music, specials, OVA, and ONA*), split between a file related to the titles (anime.csv) and one related to user ratings of the titles (training.csv). The test.csv file will be used to create the rating predictions and must be submitted for grading. The submissions.csv file illustrates the expected format of submissions.

*OVA: Original Video Animation - anime film / series made for release in home-video formats, ONA: Original Net Animation is an anime that is directly released onto the Internet.

**Supplied files:**
* anime.csv: This file contains information about the anime content, including aspects such as the id, name, genre, type, number of episodes (if applicable), an average rating based on views, and the number of members in the anime 'group'.
* train.csv:This file contains rating data, supplied by individual users for individual anime titles. It contains user_id information, the anime_id of the title watched, and the rating given (if applicable).
* test.csv: This file will be used to create the final submission. It contains a user_id and an anime_id column only - no rating (that's your task!). These ids will be used to create the rating predictions.

**Detailed file description:**

**Anime.csv**
* anime_id - myanimelist.net's unique id identifying an anime.
* name - full name of anime.
* genre - comma-separated list of genres for this anime.
* type - movie, TV, OVA, etc.
* episodes - the number of episodes in the series. (1 if movie).
* rating - average rating out of 10 for this anime.
* members - number of community members that are in this anime's "group".

**train.csv**
* user_id - non-identifiable randomly generated user id.
* anime_id - the anime that this user has rated.
* rating - rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating).

**test.csv**
* user_id - non-identifiable randomly generated user id.
* anime_id - the anime that this user has rated.

## 3. Packages <a class="anchor" id="packages"></a>

To carry out all the objectives for this repo, the following necessary dependencies were loaded:
+ `Pandas 2.2.2` and `Numpy 1.26`
+ `Matplotlib 3.8.4`
 

## 4. Environment <a class="anchor" id="environment"></a>

It's highly recommended to use a virtual environment for your projects, there are many ways to do this; we've outlined one such method below. Make sure to regularly update this section. This way, anyone who clones your repository will know exactly what steps to follow to prepare the necessary environment. The instructions provided here should enable a person to clone your repo and quickly get started.

### Create the new evironment - you only need to do this once

```bash
# create the conda environment
conda create --name <env>
```

### This is how you activate the virtual environment in a terminal and install the project dependencies

```bash
# activate the virtual environment
conda activate <env>
# install the pip package
conda install pip
# install the requirements for this project
pip install -r requirements.txt
```

## 5. Team Members<a class="anchor" id="team-members"></a>

| Name                                                                                        |  Email              
|---------------------------------------------------------------------------------------------|--------------------             
| [Team Member 1](https://github.com/<username>)                                              | team_member_email@example.com



#### Additional Resources to create a README file:
- [Make a README](https://www.makeareadme.com/)
- [GitHub Docs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
- [FreeCodeCamp](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)

