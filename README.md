# Technical Summary

This project investigated four classification algorithms for customer feedback categorisation, including KNN, Random Forest, and Neural Networks. Through our evaluation, we found that 
Random Forest worked better on majority classes, while simpler models like KNN struggled with high-dimensional and unbalanced data. After doing feature selection and tuning hyperparameters, 
the Neural Network gave the best results across all classes. It had the highest macro and weighted F1 scores among all models. In the end, we chose the Neural Network because it had better 
generalization, more stable training, and handled class imbalance more effectively. In the future, we could improve further by using ensemble methods and advanced sampling techniques like 
ADASYN to help with minority classes.

# Test Set Results and Discussion - Model Comparisons

Our group tested four different models individually and observed the following findings:

K-Nearest Neighbours (KNN):
Simple to implement but unsuitable for high-dimensional and imbalanced datasets. KNN performed reasonably well for frequent classes but failed to classify rare classes, resulting in a lower macro F1 score.

Random Forest:
Random Forest handled majority classes effectively and achieved a relatively high weighted F1 score. However, it showed a strong bias towards classes with more samples, leading to poor recall for minority classes and a low macro F1 score.


Neural Network:
When carefully tuned and combined with feature selection, the neural network achieved a better balance between frequent and rare classes. It consistently produced higher macro F1 scores on both the validation set and the test set, showing better generalisation ability.

Across all models, class imbalance remained the most significant challenge, particularly for classes with only a few samples. Models predict frequent classes better, while minority classes suffered from low recall and F1-scores.
Based on the results, the neural network gave the best weighted F1 score of 0.7186 on the validation set and 0.5320 on Test Set 2. Therefore, we decided to select the neural network model to generate predictions for both Test Set 1 and Test Set 2 submissions.
The model’s predictions successfully met all specifications with (1000×28) and (1818×28) probability matrices, though inspection revealed minor class imbalance effects in low-frequency categories. This aligns with the observed distribution shift challenge noted in Test Set 2, suggesting potential value in domain adaptation techniques for real-world deployment.
