# Credit Card Payment Default Prediction

How do banks and credit card companies plan their finances for the future? Each month, these institutions face new customers defaulting on payments. To prepare effectively, they rely on predictive models to assess individuals at risk of default. This project aims to predict whether a person will default on their upcoming credit card payment based on factors like credit limit, demographics, education level, marital status, age, and past payment history.

---

## Dataset
- **Source**: [UCI Machine Learning Repository - Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

---

## Questions and Goals

1. **Exploratory Data Analysis (EDA)**:
   - Gain insights into the dataset structure and discover patterns or trends.

2. **Primary Goal**:
   - Predict credit card payment defaults using various supervised classification algorithms.
   - Evaluate and compare the performance of these models.

3. **Clustering Analysis**:
   - Group customers into segments such as:
     - High Credit Limit, Low Risk
     - High Credit Limit, High Risk
     - Low Credit Limit, Low Risk
     - Low Credit Limit, High Risk
     - Responsible Payers vs. Delayed Payers

---

## Hypotheses

1. **Payment History**:
   - Customers who frequently failed to pay their bills over the past six months are likely to default, regardless of demographics like age or marital status.

2. **Credit Limit**:
   - Individuals with higher credit limits are less likely to default compared to those with lower credit limits.

---

## Algorithms

### **Supervised Learning Algorithms**:
- **K-Nearest Neighbors (KNN)**
- **Decision Trees / Random Forest**
- **Logistic Regression**
- **Naive Bayes**

### **Unsupervised Learning Algorithms**:
- **K-Means Clustering**:
  - Used for customer segmentation to identify behavioral patterns and risk groups.

*Note: Algorithms used may evolve as the project progresses.*

---

## Project Description

This project predicts credit card payment defaults using the **Default of Credit Card Clients Dataset**. By analyzing variables such as credit limits, demographics, and past payment behavior, the project aims to build models that help financial institutions manage risk and plan strategically.

The project involves:
- **Supervised Classification Models**: 
  - Evaluating the performance of KNN, Decision Trees/Random Forest, Logistic Regression, and Naive Bayes in predicting defaults.
- **Clustering Analysis**:
  - Leveraging K-Means Clustering to group customers based on payment behavior and risk levels.
- **Expected Outcomes**:
  - Provide actionable insights into customer behavior and help financial institutions make data-driven decisions.

---

This project combines supervised and unsupervised learning methods to provide insights into customer payment behavior and risk assessment. The models developed will enable financial institutions to predict defaults, segment customers more effectively, and make informed financial decisions.


<img width="871" alt="Screenshot 2025-01-24 at 2 04 15 AM" src="https://github.com/user-attachments/assets/a4234333-75aa-44ea-89c7-085d34db9f37" />

## Conclusion

The models had a higher accuracy on the original dataset than on the oversampled dataset. However, due to the imbalance in the dataset, the precision, recall, and F1 scores were not optimal. A model should not always be judged solely by its accuracy. Precision, recall, and F1 scores are critical metrics, and the choice of the best model depends on the context and specific business needs.

### Example: Understanding Contextual Model Selection

#### Department A: New Credit / Loan Approval
- **Prefers Precision**
  - **Why Precision Matters**:
    - **Goal**: Avoid mistakenly denying credit to individuals who are likely to pay back.
    - **Problem With Being Wrong**: Incorrectly predicting someone as a defaulter might unfairly deny them credit or loans.
    - **Solution**: Ensure high precision when predicting someone as a potential defaulter.
    - **Result**: Fewer mistakes → Fewer unfair credit denials → Happy Customers.

#### Department B: Credit Risk Management
- **Prefers Recall**
  - **Why Recall Matters**:
    - **Goal**: Identify every person likely to default to prevent financial losses.
    - **Problem With Missing Someone**: Missing a likely defaulter could lead to significant financial losses.
    - **Solution**: Focus on catching as many potential defaulters as possible.
    - **Result**: More defaulters identified → Reduced Financial Risk → Happy Stakeholders.

---

### Model Evaluation

1. **Random Forest**
   - Precision and recall significantly improved after oversampling.
   - Good overall accuracy, precision, and recall.
   - **Suitable for**: Both Department A and B.

2. **K-Nearest Neighbors (KNN)**
   - Poor recall performance on the original dataset due to class imbalance.
   - Better recall than precision after oversampling.
   - **Suitable for**: Department B.

3. **Logistic Regression**
   - Poor performance on the original dataset due to class imbalance.
   - Better recall than precision after oversampling.
   - **Suitable for**: Department B.

4. **Naive Bayes**
   - Best recall but sub-par accuracy and precision.
   - **If recall is the primary requirement**, this model is the best choice.
   - **Suitable for**: Department B.

---

### Suggestions for Improvement
The overall performance of these models can be enhanced by:
- Careful feature selection.
- Better sampling techniques.
- Hyperparameter tuning.
