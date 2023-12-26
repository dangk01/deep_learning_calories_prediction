A Multi-Stage Approach for Calorie Prediction on a New Recipe-Based Dataset (Python)

In this work, data from one of the largest online recipe websites, “AllRecipes.com” was used due to its extremely wide range of recipes, covering a multitude of cuisines from different countries and regions. Recipes on the website were scraped, obtaining their ingredients, and images Meal types, as well as the originated region/sub-regions, were collected as possible necessary features in the food recognition process. An API from Nutritionix is then queried with each recipe’s ingredients to obtain its corresponding nutritional information. The Nutritionix API stores nutrition data for restaurants, food manufacturers, and USDA’s common foods. It includes over 230K UPCs/barcodes, 100K restaurant foods, and 10,000 common foods from the USDA. We leverage our scrapped-and-prepared dataset and the Food101 dataset (described in sections 3.1.4 and 3.2.2) and apply different deep-learning architectures to accomplish nutrient estimation tasks. Our propose is to consider this task as a regression problem and utilize convolutional neural networks (CNNs) models for image recognition with Huber Loss Function to make predictions so that the mean absolute error is minimized. To achieve better performance, we suggest another approach to use a pre-trained model to retrieve the estimated ”weights” of specific food classes, which then can produce a better prediction of the amounts of calories, protein, fat, and carbohydrates

# Team:
- Khoa Dang (tkdang97@gmail.com)
- Eman Wong (emanwong@umich.edu)
