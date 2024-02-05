from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, df):
        # Initialize the preprocessor with a DataFrame
        self.df = df

    def preprocess(self):
        # Apply various preprocessing methods on the DataFrame
        self.df = self._preprocess_imputation(self.df)
        self.df = self._preprocess_numerical(self.df)
        self.df = self._preprocess_categorical(self.df)
        self.df = self._preprocess_ordinal(self.df)
        return self.df

    def _preprocess_imputation(self,df): 
        # Using mode to imputate
        cnt_col = 0
        for col in df.columns:
            cnt_col+=1
            sum, denominator = 0, 0
            nan_index, t_index, f_index = [], [], []
            for index, row in df.iterrows():
                if cnt_col <= 17:
                    if pd.isna(row[col]):
                        nan_index.append(index)
                    else: 
                        sum+=row[col]
                        denominator+=1
                else:
                    if pd.isna(row[col]):
                        nan_index.append(index)
                    else:
                        if row[col] == 1:
                            t_index.append(index)
                        else:
                            f_index.append(index)
            if cnt_col <= 17:
                avg = round(sum/denominator, 2)
                for index in nan_index:
                    df.at[index,col] = avg
            elif cnt_col <= 77:
                if len(t_index) > len(f_index):
                    for index in nan_index:
                        df.at[index,col] = 1
                else: 
                    for index in nan_index:
                        df.at[index,col] = 0

        # print("preprocess imputation:")
        # print(df.head())
        return df

    def _preprocess_numerical(self, df):
        # Custom logic for preprocessing numerical features goes here
        cnt_col = 0
        for col in df.columns:
            cnt_col+=1
            if cnt_col > 17: break
            maxx,minn = max(df[col]), min(df[col])
            # print(f"feature:{cnt_col} max: {maxx} min: {minn}")
            for index, row in df.iterrows():
                df.at[index, col] = round((row[col] - minn)/(maxx - minn), 4)
        # print("preprocess numeric:")
        # print(df.head())
        return df

    def _preprocess_categorical(self, df):
        # Add custom logic here for categorical features
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_features:
            df[col] = LabelEncoder().fit_transform(df[col])
        return df

    def _preprocess_ordinal(self, df):
        # Custom logic for preprocessing ordinal features goes here
        return df

# Implementing the classifiers (NaiveBayesClassifier, KNearestNeighbors, MultilayerPerceptron)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        # Abstract method to fit the model with features X and target y
        pass
    @abstractmethod
    def predict(self, X):
        # Abstract method to make predictions on the dataset X
        pass
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Naive Bayes Classifier
class NaiveBayesClassifier(Classifier):
    def __init__(self):
        # Initialize the probability table for categorical features. feature_result -> true_true
        self.class_prior_true_true = []
        self.class_prior_true_false = []
        self.class_prior_false_true = []
        self.class_prior_false_false = []

        # Initialize the probability table for continuous features. feature_result -> true_true
        self.conti_mean_true = []
        self.conti_SD_true = []
        self.conti_mean_false = []
        self.conti_SD_false = []

        # Different between data set
        self.conti_feature_start = 0
        self.conti_feature_end = 16
        self.class_feature_start = 17
        self.class_feature_end = 76
        # Calculate how many true false cases in training dataset
        self.true_cases = 0
        self.false_cases = 0

    def fit(self, X, y):
        # Implement the fitting logic for Naive Bayes classifier

        # Continuous features
        for m, column_name in enumerate(X.columns):
            tmp1, tmp2 = [], []
            for n, row in X.iterrows():
                if y[n] == 1: tmp1.append(X[column_name][n])
                else: tmp2.append(X[column_name][n])
            
            self.conti_mean_true.append(np.mean(tmp1))
            self.conti_SD_true.append(np.std(tmp1))
            self.conti_mean_false.append(np.mean(tmp2))
            self.conti_SD_false.append(np.std(tmp2))
            if m >= self.conti_feature_end : break

        # Categorical features
        for m, column_name in enumerate(X.columns):
            tmp1, tmp2, tmp3, tmp4, cnt1, cnt2 = 0, 0, 0, 0, 0, 0
            if m >= self.class_feature_start:
                for n, row in X.iterrows():
                    if y[n] == 1:
                        cnt1 += 1
                        if X[column_name][n] == 1: tmp1 += 1
                        else: tmp2 += 1 
                    else: 
                        cnt2+=1
                        if X[column_name][n] == 1: tmp3 += 1
                        else: tmp4 += 1  
                self.class_prior_true_true.append(tmp1 / cnt1)
                self.class_prior_false_true.append(tmp2 / cnt1)
                self.class_prior_true_false.append(tmp3 / cnt2)
                self.class_prior_false_false.append(tmp4 /cnt2)
                self.true_cases = cnt1
                self.false_cases = cnt2


    def predict(self, X):
        # Implement the prediction logic for Naive Bayes classifier
        y = []
        for m, row in X.iterrows():
            truee, falsee = 1, 1
            # Calculate the prob of true
            for n, column_name in enumerate(X.columns):
                if n <= self.conti_feature_end:
                    truee *= self.Normal_distribution(n, X[column_name][m], T = True)
                else:
                    if row[column_name] == 1: 
                        truee *= self.class_prior_true_true[n - self.class_feature_start]
                    else: 
                        truee *= self.class_prior_false_true[n - self.class_feature_start]

            # Calculate the prob of false
            for n, column_name in enumerate(X.columns):
                if n <= self.conti_feature_end:
                    falsee *= self.Normal_distribution(n, X[column_name][m], T = False)
                else:
                    if row[column_name] == 1: 
                        falsee *= self.class_prior_true_false[n - self.class_feature_start]
                    else: 
                        falsee *= self.class_prior_false_false[n - self.class_feature_start]

            truee *= self.true_cases
            falsee *= self.false_cases

            if truee >= falsee: 
                y.append(np.int64(1))
            else:
                y.append(np.int64(0))

        y = np.array(y)
        # print(y)
        return y
        
    def predict_proba(self, X):
        # Implement probability estimation for Naive Bayes classifier
        prob = []
        for m, row in X.iterrows():
            truee, falsee = 1, 1
            # Calculate the prob of true
            for n, column_name in enumerate(X.columns):
                if n <= self.conti_feature_end: 
                    truee *= self.Normal_distribution(n, X[column_name][m], T = True)
                else:
                    if row[column_name] == 1: 
                        truee *= self.class_prior_true_true[n - self.class_feature_start]
                    else: 
                        truee *= self.class_prior_false_true[n - self.class_feature_start]

            # Calculate the prob of false
            for n, column_name in enumerate(X.columns):
                if n <= self.conti_feature_end: 
                    falsee *= self.Normal_distribution(n, X[column_name][m], T = False)
                else:
                    if row[column_name] == 1: 
                        falsee *= self.class_prior_true_false[n - self.class_feature_start]
                    else: 
                        falsee *= self.class_prior_false_false[n - self.class_feature_start]

            truee *= self.true_cases
            falsee *= self.false_cases

            if truee >= falsee: 
                prob.append([truee/(truee + falsee), falsee/(truee + falsee)])
            else:
                prob.append([falsee/(truee + falsee), truee/(truee + falsee)])

        prob = np.array(prob) 
        # print(prob)
        return prob

    def Normal_distribution(self, index, value, T):
        # I eliminate "np.sqrt(2 * np.pi)" to lower the truncation error
        if T:
            part1 = 1 / self.conti_SD_true[index]
            part2 = np.exp(-((value - self.conti_mean_true[index])**2) / (2 * self.conti_SD_true[index]**2)) 
        else:
            part1 = 1 / self.conti_SD_false[index]
            part2 = np.exp(-((value - self.conti_mean_false[index])**2) / (2 * self.conti_SD_false[index]**2)) 

        # print(part1*part2)
        return part1 * part2 

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class KNearestNeighbors(Classifier):
    def __init__(self, k = 11):
        self.k = k
        self.model = None
        self.p = 2

    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        self.model = data

    def predict(self, X):
        y = []
        for index, row in X.iterrows():
            minkowski_dis = np.power(np.sum(np.power(np.abs(self.model.iloc[:, :77] - row), self.p), axis=1), 1/self.p)
            sorted_indices = np.argsort(minkowski_dis).tolist() # You need to turn it into a original array 

            true_num, false_num = 0, 0
            for candidate in range(self.k):
                if self.model.iloc[sorted_indices[candidate], -1] == 1:  true_num += 1
                else:  false_num += 1

            if true_num > false_num:
                y.append(np.int64(1))
            else:
                y.append(np.int64(0))

        return y

    def predict_proba(self, X):
        # Implement probability estimation for KNN
        prob = []
        for index, row in X.iterrows():
            minkowski_dis = np.power(np.sum(np.power(np.abs(self.model.iloc[:, :77] - row), self.p), axis=1), 1/self.p)
            sorted_indices = np.argsort(minkowski_dis).tolist() # You need to turn it into a original array 

            true_num, false_num = 0, 0
            for candidate in range(self.k):
                if self.model.iloc[sorted_indices[candidate], -1] == 1:  true_num += 1
                else:  false_num += 1

            if true_num > false_num:
                prob.append([(true_num/self.k), (false_num/self.k)])
            else:
                prob.append([(false_num/self.k), (true_num/self.k)])

        prob = np.array(prob)
        return prob
    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Single perceptron, which output size is always 1
class SingleNeuronPerceptron(Classifier):
    def __init__(self, input_size, output_size):
        # Initialize single neuron 
        '''
        self.weights = np.random.rand(input_size)/np.sqrt(input_size)
        '''
        # Initialize single neuron with Xavier initializaiton
        self.weights = np.random.rand(input_size)/np.sqrt(input_size)
        self.output_size = output_size
        self.bias = 1 
        self.learning_rate = 0.2
        self.epochs = 1500
        
    def fit(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for index, row in X.iterrows():
                output = self._forward_propagation(row)
                self._backward_propagation(row, output, y[index])
                total_error += 0.5 * (y[index] - output) ** 2
            print(f"Epoch {epoch + 1}/{self.epochs}, Error: {total_error}")

    def predict(self, X):
        y = []
        for index, row, in X.iterrows():
            target = self._forward_propagation(row)
            if target >= 0.5: target = 1
            else: target = 0 
            y.append(target)

        y = np.array(y)
        return y

    def predict_proba(self, X):
        prob = []
        for index, row in X.iterrows():
            target = self._forward_propagation(row)
            if target >= 0.5: 
                prob.append([target, 1 - target])
            else: 
                prob.append([1 - target, target])

        prob = np.array(prob)
        return prob
        
    def _forward_propagation(self, X):
        weighted_sum = np.dot(X, self.weights) + self.bias
        output = self.sigmoid(weighted_sum)
        return output

    def _backward_propagation(self, input, output, target):
        error = target - output
        gradient = error * self.sigmoid(output, derivative = True)
        self.weights += self.learning_rate * gradient * np.array(input)
        self.bias += self.learning_rate * gradient

    # Activation Funciton 
    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Output size in this task is always 1
class MultilayerPerceptron(Classifier):
    def __init__(self, input_size, hidden_layers_sizes, output_size):
        # Initialize MLP with given network structure
        self.hidden_layers_sizes = hidden_layers_sizes
        # Normal way to initialize weights (Cause tremendous total error) 
        '''
        # self.weights = [np.random.rand(input_size, hidden_layers_sizes[0])] # matrix dimension: inputsize * hidden_layer[0] 
        # self.weights += [np.random.rand(hidden_layers_sizes[i], hidden_layers_sizes[i+1]) for i in range(len(hidden_layers_sizes)-1)]
        # self.weights += [np.random.rand(hidden_layers_sizes[-1], output_size)]
        '''
        # Using Xavier/Glorot initialization to initialize weights 
        self.weights = [np.random.randn(input_size, hidden_layers_sizes[0]) / np.sqrt(input_size)]  
        self.weights += [np.random.randn(hidden_layers_sizes[i], hidden_layers_sizes[i+1]) / np.sqrt(hidden_layers_sizes[i]) for i in range(len(hidden_layers_sizes)-1)]
        self.weights += [np.random.randn(hidden_layers_sizes[-1], output_size) / np.sqrt(hidden_layers_sizes[-1])]

        self.biases = [np.ones(hidden_size) for hidden_size in hidden_layers_sizes] # Every neuron has one bias
        self.biases += [np.ones(output_size)] # Output size in this task is always one
        self.delta = [np.random.rand(hidden_layers_sizes[i]) for i in range(len(hidden_layers_sizes))] #The val will be flush during backward_prop 
        self.delta += [np.random.rand(output_size)] # Output size in this task is always one
        self.inner_output = [np.random.rand(hidden_layers_sizes[i]) for i in range(len(hidden_layers_sizes))] #The val will be flush during forward_prop 
        self.inner_output += [np.random.rand(1)] # This val is equal to the final Ouput

        self.learning_rate = 0.2
        self.epochs = 1500


    def fit(self, X, y):
        # Implement training logic for MLP including forward and backward propagation
        for epoch in range(self.epochs):
            total_error = 0
            for index, row in X.iterrows():
                output = self._forward_propagation(row)
                self._backward_propagation(row, output, y[index])
                total_error += 0.5 * (y[index] - output) ** 2
            print(f"Epoch {epoch + 1}/{self.epochs}, Error: {total_error[0]}")

    def predict(self, X):
        # Implement prediction logic for MLP
        y = []
        for index, row in X.iterrows():
            target = self._forward_propagation(row)
            if target >= 0.5: target = 1
            else: target = 0
            y.append(target)

        y = np.array(y)
        print(y)
        return y

    def predict_proba(self, X):
        # Implement probability estimation for MLP
        prob = []
        for index, row in X.iterrows():
            target = self._forward_propagation(row)
            if target >= 0.5: 
                prob.append([(target), (1 - target)])
            else: 
                prob.append([(1 - target), (target)])

        prob = np.array(prob)
        return prob

    def _forward_propagation(self, X):
        # Implement forward propagation for MLP
        layer_output = X
        for i in range(len(self.weights)):
            weighted_sum = np.dot(layer_output, self.weights[i]) + self.biases[i] # Notice the size of the weights 
            layer_output = self.sigmoid(weighted_sum)
            self.inner_output[i] = layer_output

        return layer_output

    def _backward_propagation(self, input, output, target):
        # Implement backward propagation for MLP
        # Ouput Unit Weights 
        error = target - output 
        self.delta[len(self.weights)-1] = self.sigmoid(output, derivative=True) * error # ouput == self.inner_output[len(self.weights)-1]
        self.weights[len(self.weights)-1] += self.learning_rate * (self.inner_output[len(self.weights)-2][:, np.newaxis] * self.delta[len(self.weights)-1][np.newaxis, :])

        # Hidden Unit Weights
        for layer in range(len(self.weights) - 2, 0, -1):
            self.delta[layer] = np.multiply(self.sigmoid(self.inner_output[layer], derivative=True), np.dot(self.weights[layer + 1], self.delta[layer + 1]))
            self.weights[layer] += self.learning_rate * (self.inner_output[layer - 1][:, np.newaxis] * self.delta[layer][np.newaxis, :])


        # First layer of the hidden layer
        self.delta[0] = np.multiply(self.sigmoid(self.inner_output[0], derivative=True), np.dot(self.weights[1], self.delta[1]))
        self.weights[0] += self.learning_rate * (np.array(input)[:, np.newaxis] * self.delta[0][np.newaxis, :])


        # Update biases
        for i in range(len(self.biases)):
            self.biases[i] += self.learning_rate * self.delta[i]

    # Activation Funciton 
    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Function to evaluate the performance of the model
def evaluate_model(model, X_test, y_test):
    # Predict using the model and calculate various performance metrics
    predictions = model.predict(X_test)
    # print(y_test)
    # print(predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions) 
    recall = recall_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)

    # Check if the model supports predict_proba method for AUC calculation
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:  # Binary classification
            auc = roc_auc_score(y_test, proba[:, 1])
        else:  # Multiclass classification
            auc = roc_auc_score(y_test, proba, multi_class='ovo')
    else:
        auc = None

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'auc': auc
    }
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''   
# Main function to execute the pipeline
def main():
    # Load trainWithLable data
    df = pd.read_csv('Dataset/trainWithLabel.csv')

    # Preprocess the training data --> Wrong code  
    # Split the dataset into features and target variable --> Wrong code 
    '''
    preprocessor = Preprocessor(df)
    df_processed = preprocessor.preprocess()
    X_train = df_processed.drop('Outcome', axis=1)
    y_train = df_processed['Outcome']
    '''

    # Define the models for classification
    models = {'Naive Bayes': NaiveBayesClassifier(), 
              'KNN': KNearestNeighbors(), 
              'SNP' :SingleNeuronPerceptron(77, 1),
              'MLP': MultilayerPerceptron(77, (8, 4), 1) # size of hidden layers should be at least 1! -> (5,3) is 2 hidden layers
    }

    X_train = df.drop('Outcome', axis=1)
    y_train = df['Outcome']

    # Perform K-Fold cross-validation
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = []

    for name, model in models.items():
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train), start=1):
            # Slightly modify here
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            # We should do preprocess here
            processed = Preprocessor(X_train_fold)
            X_train_fold = processed.preprocess()
            processed = Preprocessor(X_val_fold)
            X_val_fold = processed.preprocess()

            model.fit(X_train_fold, y_train_fold)
            fold_result = evaluate_model(model, X_val_fold, y_val_fold)
            fold_result['model'] = name
            fold_result['fold'] = fold_idx
            cv_results.append(fold_result)

    # Convert CV results to a DataFrame and calculate averages
    cv_results_df = pd.DataFrame(cv_results)
    avg_results = cv_results_df.groupby('model').mean().reset_index()
    avg_results['model'] += ' Average'
    all_results_df = pd.concat([cv_results_df, avg_results], ignore_index=True)

    # Adjust column order and display results
    all_results_df = all_results_df[['model', 'accuracy', 'f1', 'precision', 'recall', 'mcc', 'auc']]

    print("Cross-validation results:")
    print(all_results_df)

    # Save results to an Excel file
    all_results_df.to_excel('Results/cv_results.xlsx', index=False)
    print("Cross-validation results with averages saved to cv_results.xlsx")

    # Load the test dataset, assuming you have a test set CSV file without labels
    df_ = pd.read_csv('Dataset/testWithoutLabel.csv')
    preprocessor_ = Preprocessor(df_)
    X_test = preprocessor_.preprocess()

    # Initialize an empty list to store the predictions of each model
    predictions = []

    # Make predictions with each model
    for name, model in models.items():
        model_predictions = model.predict(X_test)
        predictions.append({
            'model': name,
            'predictions': model_predictions
        })

    # Convert the list of predictions into a DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Print the predictions
    print("Model predictions:")
    print(predictions_df)

    # Save the predictions to an Excel file
    predictions_df.to_csv('Results/test_results.csv', index=False)
    print("Model predictions saved to test_results.xlsx")

if __name__ == "__main__":
    main()
