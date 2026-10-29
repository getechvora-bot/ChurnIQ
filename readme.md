## Model Comparison

Three classification models were evaluated for customer churn prediction using the same test set of 1,409 customers.

| Model | Test Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 0.6040 |
| Default Decision Tree | 72.25% | 47.81% | 49.73% | 0.4875 |
| Tuned Decision Tree | 79.77% | 64.98% | 51.60% | 0.5753 |

## Key Findings

- Logistic Regression achieved the highest test accuracy and churn F1-score among the three models.
- Logistic Regression achieved a churn recall of 55.88%, identifying 209 of the 374 actual churn customers in the test set.
- The default Decision Tree had a large train-test accuracy gap, indicating likely overfitting.
- Hyperparameter tuning improved the Decision Tree's test performance compared with the default Decision Tree.
- The tuned Decision Tree achieved better churn precision and recall than the default tree, but its churn F1-score remained below Logistic Regression's.

## Conclusion

Logistic Regression is the preferred model among the three evaluated models based on the current test-set results. However, it missed 165 of the 374 actual churn customers. Before deployment, further analysis should consider the business cost of missed churn customers, false alarms, probability thresholds, and validation on suitable new data.

## Limitations

- Results depend on the dataset, train-test split, preprocessing, and evaluation method.
- Test accuracy alone is not sufficient for choosing a churn model.
- Churn precision, recall, F1-score, and the costs of false positives and false negatives should also be considered.
- Performance on this test set does not guarantee performance on future customers.
