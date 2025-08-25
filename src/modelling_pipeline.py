"""
Model fitting, tuning and evaluation pipeline.

Future considerations:
* Candidate models to include KNN, SGD, SVM, XGBoost
* SHAP
"""

# load libraries and modules
import pandas as pd
import numpy as np, warnings
from pathlib import Path
import os
from importlib import reload

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# data processing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel

# model comparison
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Evaluation metrics
from sklearn.metrics import cohen_kappa_score,classification_report 
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, auc, make_scorer
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, balanced_accuracy_score, PrecisionRecallDisplay
import shap


class ModelPipeline:

    # Init parameters
    ROOT_DIR = Path('../..')
        
    def __init__(self, ROOT_DIR):
        self.ROOT_DIR = ROOT_DIR
        # plot output
        self.output_dir = self.ROOT_DIR / 'output' / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_dfs(self, tops):
        dfs_dict = {}
        for top in tops:
            dfs_dict[f"top{top}"] = self.get_model_input_dfs(f'top{top}')
        return dfs_dict


    def get_full_dfs(self, tops):
        dfs_dict = {}
        for top in tops:
            dfs_dict[f"top{top}"] = self.get_model_full_dfs(f'top{top}')
        return dfs_dict


    def get_model_input_dfs(self, top, timepoints=['t0', 't1', 't2', 't3', 't4']):
        
        """
        Function to read the model input dataframes based on provided prefixes and date string.
        
        Parameters:
        - ROOT_DIR: The root directory path.
        - prefix: The prefix (e.g. 'top20', 'top30', etc.)
        - timepoints: A list of timepoints (default is ['t0', 't1', 't2', 't3', 't4']).
        
        Returns:
        - A dictionary with keys as timepoints and values as the corresponding dataframes.
        """
        
        dfs = {}
        for tp in timepoints:
            path = self.ROOT_DIR / 'data' / 'Model input data' / f"{tp}_{top}.csv"
            dfs[tp] = self.get_model_input_df(path)
        
        return dfs


    def get_model_full_dfs(self, top, timepoints=['t0', 't1', 't2', 't3', 't4']):
        
        """
        Function to read the model input dataframes based on provided prefixes and date string.
        
        Parameters:
        - ROOT_DIR: The root directory path.
        - prefix: The prefix (e.g. 'top20', 'top30', etc.)
        - timepoints: A list of timepoints (default is ['t0', 't1', 't2', 't3', 't4']).
        
        Returns:
        - A dictionary with keys as timepoints and values as the corresponding dataframes.
        """
        
        dfs = {}
        for tp in timepoints:
            path = self.ROOT_DIR / 'data' / 'full_data' / f"{tp}_{top}.csv"
            dfs[tp] = self.get_model_input_df(path)
        
        return dfs


    def get_model_input_df(self, model_input_path):
        """_summary_
        
        Args:
            model_input_path (_type_): Processed csv file path.
        """
        model_input_df = pd.read_csv(model_input_path)
        model_input_df = model_input_df.drop(columns=["SUBJECT_ID", "HADM_ID"])
        return model_input_df


    def count_missing_values(self, time_df):
        """
        Get count of missing values.
        
        Args:
            time_df (_type_): Processed csv file path.
        """
        for time_point, df in time_df.items():
            df_without_target = df.drop(columns=['IS_SEPSIS'])
            total_values = df_without_target.size
            count_999 = (df_without_target == -999).sum().sum()
            missing_proportions = count_999 / total_values *100
            print(f"Number of missing values in {time_point}: {count_999} ({missing_proportions:.2f}%)")

    def count_missing_values_all(self, dfs_dict, hours_list, top_n_features, sepsis_only = False):
        """
        Get count of missing values for all DF in dictionary. 
                
        Args:
            dfs_dict (Dict): Dictionary of data extractions at different time points.
            hours_list (_type_): _description_
            top_n_features (_type_): _description_
            sepsis_only: Only account for Sepsis admissions or All admissions

        Returns:
            result_df (DataFrame): NA Counts and NA Proportions
        """
        # Initialise dataframes to store results
        na_counts = pd.DataFrame(index=hours_list, columns=[f'top{n}_features_NA_Count' for n in top_n_features])
        na_proportions = pd.DataFrame(index=hours_list, columns=[f'top{n}_features_NA_%' for n in top_n_features])

        # Iterate through input dictionary
        for top_n_key, time_dfs in dfs_dict.items():
            for time_point, df in time_dfs.items():
                if sepsis_only == True:
                    df = df[df['IS_SEPSIS'] == 1]
                    
                df_without_target = df.drop(columns=['IS_SEPSIS'])
                total_values = df_without_target.size
            
                # count missing
                count_999 = (df_without_target == -999).sum().sum()
                missing_proportions = count_999 / total_values * 100
                formatted_missing_proportions = "{:.2f}%".format(missing_proportions)

                # Extract the hour and top_n from the keys
                hour = int(time_point.lstrip('t'))  # remove 't' and convert to int
                top_n = int(top_n_key.lstrip('top'))  # remove 'top' and convert to int
                
                # Store results in dataframes
                na_counts.loc[hour, f'top{top_n}_features_NA_Count'] = count_999
                na_proportions.loc[hour, f'top{top_n}_features_NA_%'] = formatted_missing_proportions

        result_df = pd.DataFrame(index=hours_list)

        # Concatenate the data for each top_n features
        for top_n in top_n_features:
            na_count_col = f'top{top_n}_features_NA_Count'
            na_prop_col = f'top{top_n}_features_NA_%'
            result_df[na_count_col] = na_counts[na_count_col]
            result_df[na_prop_col] = na_proportions[na_prop_col]

        result_df.index = ['t' + str(i) for i in result_df.index]

        return result_df
        
        #return na_counts, na_proportions

    def count_missing_SOFA(self, dfs_dict, hours_list, sepsis_only=False):
        """
        Get count of missing SOFA for all DF in dictionary. 
                
        Args:
            dfs_dict (Dict): Dictionary of data extractions at different time points.
            hours_list (_type_): _description_
            sepsis_only (bool): Only account for Sepsis admissions or All admissions

        Returns:
            result_df (DataFrame): NA Counts and NA Proportions
        """
        # Initialise dataframes to store results
        na_counts = pd.DataFrame(index=hours_list, columns=['SOFA_NA_Count'])
        na_proportions = pd.DataFrame(index=hours_list, columns=['SOFA_NA_%'])

        # Iterate through input dictionary
        for _, time_dfs in dfs_dict.items():
            for time_point, df in time_dfs.items():
                if sepsis_only:
                    df = df[df['IS_SEPSIS'] == 1]
                    
                total_SOFA = df.shape[0]
            
                # count missing SOFA values
                count_SOFA_999 = (df['SOFA'] == -999).sum()
                missing_proportions = count_SOFA_999 / total_SOFA * 100
                formatted_missing_proportions = "{:.2f}%".format(missing_proportions)

                # Extract the hour
                hour = int(time_point.lstrip('t'))  # remove 't' and convert to int
                
                # Store results in dataframes
                na_counts.loc[hour, 'SOFA_NA_Count'] = count_SOFA_999
                na_proportions.loc[hour, 'SOFA_NA_%'] = formatted_missing_proportions

        result_df = pd.DataFrame(index=hours_list)
        result_df['SOFA_NA_Count'] = na_counts['SOFA_NA_Count']
        result_df['SOFA_NA_%'] = na_proportions['SOFA_NA_%']
        result_df.index = ['t' + str(i) for i in result_df.index]

        return result_df

    def split_data(self, df_train):
        """
        Split data into training and test sets.
        Standardise numerical features.
        
        Compare performances of different models.
        Perform hyperparameter tuning on best model.
        Validate the optimal model's performance on the test set.
        Generate predictions for unknown data using the optimal model.
        
        Parameters:
        - df_train: Processed data for model training
        - test_size: 
        
        Returns:
        - X train, X test, y train, y test 
        """
            
        # Target and Predictors
        X = df_train.drop('IS_SEPSIS', axis='columns')
        y = df_train['IS_SEPSIS']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Fit the scaler and transform the X train and test sets
        # Standardising (not normalising!)
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test


    def get_class_weights(self, y_train):
        """
            Compute class weights for balancing the classes in the target variable.

        Args:
            y_train (_type_)

        Returns:
            class_weights (dict)
        """
        # Get sepsis label proportions
        label_counts = y_train.value_counts()
        label_proportions = label_counts / len(y_train)*100
        
        # Compute class weights
        class_weights = {0: 1 / (label_proportions[0] / 100), 1: 1 / (label_proportions[1] / 100)}

        # Round the class weights to the desired precision (optional)
        class_weights = {key: round(weight, 4) for key, weight in class_weights.items()}
        
        return class_weights


    def static_models(self, candidate_models, class_weights, X_train, y_train, X_test, y_test, time):
        '''
        Fit static models.
        
        Returns:
            pd.DataFrame: A DataFrame containing model names and their average cross-validation scores.
        
        '''
        
        model_names = list(candidate_models.keys())
        
        performance_df = pd.DataFrame(columns=['Model', 'Balanced_Acc_Train', 'Balanced_Acc_Test', 'Precision_Train', 'Precision_Test', 'Recall_Train', 'Recall_Test', 'F1_Train', 'F1_Test', 'AUROC_Train', 'AUROC_Test'])

        for model_name, model in candidate_models.items():

            y_pred_train, y_pred_test = self.fit_models(model, model_name, class_weights, X_train, y_train, X_test)

            performance_scores = self.get_performance_scores(model, model_name, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test)

            formatted_model_name = f"{model_name}_{time}"
            new_row = pd.DataFrame([[formatted_model_name] + performance_scores], columns=performance_df.columns)

            performance_df = pd.concat([performance_df, new_row], ignore_index=True)

            predicted_probabilities = model.predict_proba(X_test)

        return performance_df

    def perform_cv_for_conditions(self, hours_list, top_n_features, split_data_dict, candidate_models, all_models=True, LR_only=False, RF_only=False, GB_only=False):
        """
        Performs cross-validation for given conditions and data.
        
        Args:
            conditions (list of tuples): List of (top_features, time_window) conditions.
            split_data_dict (dict): Dictionary with data splits corresponding to conditions.
            models (dict): Candidate models to perform CV.
            mp (Module): Module that contains the cv_analysis function.
        
        Returns:
            pd.DataFrame: Consolidated results of CV for the conditions.
        """
        all_results = []

        for hour in hours_list:
            for top_n in top_n_features:    
                current_data = split_data_dict[(top_n, hour)]
                X_train = current_data['X_train']
                y_train = current_data['y_train']
                
                current_results = self.cv_analysis(X_train, y_train, candidate_models, all_models, LR_only, RF_only, GB_only)
                
                current_results['top_features'] = top_n
                current_results['time_window'] = hour
                
                all_results.append(current_results)

        return pd.concat(all_results, ignore_index=True)

    def cv_analysis(self, X_train, y_train, candidate_models, all_models=True, LR_only=False, RF_only=False, GB_only=False):
        """
        Perform cross-validation analysis on a set of candidate models and return their mean scores.

        Args:
            X_train (pd.DataFrame): Scaled feature matrix for training.
            y_train (pd.Series): Target labels for training.
            candidate_models (Dict[str, Any]): A dictionary of static ML model names and their respective instantiated models.
            
        Returns:
            pd.DataFrame: A DataFrame containing model names and their training and cross-validation average balanced accuracy scores.
        """
        
        model_names = []
        model_average_scores_cv = [] 
        model_average_scores_train = []
        
        # Determine which models to evaluate
        models_to_evaluate = []
        if all_models:
            models_to_evaluate = list(candidate_models.keys())
        if LR_only:
            models_to_evaluate.append('Logistic_Regression')
        if RF_only:
            models_to_evaluate.append('Random_Forest')
        if GB_only:
            models_to_evaluate.append('Gradient_Boosting')
        
        # Calculate mean scores using cross validation and training set
        for model_name in models_to_evaluate:
            model = candidate_models.get(model_name)
            if not model:
                continue  # Skip if the model is not in the candidate models
            
            # Cross-validation score
            scores_cv = cross_val_score(model, X_train, y_train, scoring='balanced_accuracy')
            model_average_scores_cv.append(scores_cv.mean())
            
            # Training score
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            score_train = balanced_accuracy_score(y_train, y_train_pred)
            model_average_scores_train.append(score_train)
            
            model_names.append(model_name)
                
        # Store mean scores for each model
        df_model = pd.DataFrame({
            'model': model_names,
            'train_balanced_acc': model_average_scores_train,
            'cv_balanced_acc': model_average_scores_cv
        })
        
        return(df_model)

    
    def cv_analysis_all(self, X_t0_train, y_t0_train, 
                                X_t1_train, y_t1_train,
                                X_t2_train, y_t2_train,
                                X_t3_train, y_t3_train,
                                X_t4_train, y_t4_train, candidate_models, hours_list, top_n_features):
        """
        Perform cross-validation analysis on a set of candidate models and return their mean scores.

        Args:
            X_train (pd.DataFrame): Scaled feature matrix for training.
            y_train (pd.Series): Target labels for training.
            candidate_models (Dict[str, Any]): A dictionary of static ML model names and their respective instantiated models.
            class_weights (Dict[int, float]): A dictionary of class weights for handling sepsis class imbalance.
            
        Returns:
            pd.DataFrame: A DataFrame containing model names and their average cross-validation scores.
        """
        
        results = [] 
        
        for hour in hours_list:
            X_train = eval(f"X_t{hour}_train")
            y_train = eval(f"y_t{hour}_train")

            for top_n in top_n_features:
                # Calculate mean scores using cross validation
                for model_name, model in candidate_models.items():
                    scores = cross_val_score(model, X_train, y_train, scoring = 'balanced_accuracy', cv=3)
                    average_score = scores.mean()
                    results.append({
                        'model': f"{model_name}_{top_n}",
                        f't{hour}': average_score
                    })
                    
        df_model = pd.DataFrame(results)
                
        # Calculate overall average balanced accuracy for each model
        df_model['overall_avg_balanced_acc'] = df_model[['t0', 't1', 't2', 't3', 't4']].mean(axis=1)

        return(df_model)

    def fit_models(self, model, model_name, class_weights, X_train, y_train, X_test):
        """_summary_
        Model training function (inside models_fitting)
        
        Args:
            model (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        if model_name == 'Gradient_Boost':
            model.fit(X_train, y_train, sample_weight=[class_weights[label] for label in y_train])
        else:
            model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        return y_pred_train, y_pred_test



    def get_performance_scores(self, model, model_name, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test):
        """
        Get performance measures on (i) ballanced accuracy (ii) precision, (iii) recall and (iv) F1 score.
        
        Called by models_fitting()
        
        Args:
            model (_type_): _description_
            X_train (_type_): Scaled X_train
            X_test (_type_): Scaled X_train
            y_train (_type_): _description_
            y_test (_type_): _description_
        
        Returns:
        """
        
        # Computing balanced accuracy
        balanced_acc_train = balanced_accuracy_score(y_train, y_pred_train)
        balanced_acc_test = balanced_accuracy_score(y_test, y_pred_test)

        # Computing precision and recall
        precision_train = precision_score(y_train, y_pred_train)
        precision_test = precision_score(y_test, y_pred_test)
        recall_train = recall_score(y_train, y_pred_train)
        recall_test = recall_score(y_test, y_pred_test)

        # Computing F1 score
        f1_train = f1_score(y_train, y_pred_train)
        f1_test = f1_score(y_test, y_pred_test)

        # Compute the probability scores of the positive class
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]

        # Compute AUC
        auc_train = roc_auc_score(y_train, y_train_prob)
        auc_test = roc_auc_score(y_test, y_test_prob)

        # Format scores
        performance_scores = [balanced_acc_train, balanced_acc_test, precision_train, precision_test, recall_train, recall_test, f1_train, f1_test, auc_train, auc_test]
        
        # formatted_performance_scores = []
        # for i in range(len(performance_scores)):
        #   s = '({i:.4f})'.format(performance_scores[i])
        #   formatted_performance_scores.append(s)
        
        # return formatted_performance_scores
        return performance_scores


    def plot_results(self, candidate_models, class_weights, X_train, X_test, y_train, y_test):
        """
        Get model plots.
        
        Args:
            candidate_models (dict): _description_
            X_train (_type_): Scaled X_train
            X_test (_type_): Scaled X_train
            y_train (_type_): Scaled Y_test
            y_test (_type_): Scaled Y_test
        
        Returns:
        """

        for model_name, model in candidate_models.items():

            y_pred_train, y_pred_test = self.fit_models(model, model_name, class_weights, X_train, y_train, X_test)
            predicted_probabilities = model.predict_proba(X_test)

            self.plot_confusion_matrix(model, model_name, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test)
            self.plot_precision_recall(model, model_name, X_test, y_test)
            self.plot_roc_curve(predicted_probabilities, model_name, y_test, label = None)


    def plot_combined_roc_curves(self, candidate_models, X_test, y_test, X_test_lstm = [], y_test_lstm = [], title='ROC Curves'):
        """
        Plot the ROC curve for the candidate models.
        

        """

        # Initialise the plot
        plt.figure(figsize=(6, 6))
        
        # Iterate over classifiers and plot ROC curve for each
        for model_name, model in candidate_models.items():
            if model_name == 'LSTM':
                # Predict the probabilities of the positive class
                predicted_probabilities_lstm = model.predict(X_test_lstm)
                
                # Compute ROC curve and AUC
                fpr_lstm, tpr_lstm, thresholds_roc_test_lstm = roc_curve(y_test_lstm[:, 1], predicted_probabilities_lstm[:, 1], pos_label=1)
                roc_auc_lstm = auc(fpr, tpr)# Plot ROC curve
                
                # Plot ROC curve
                plt.plot(fpr_lstm, tpr_lstm, label=f'{model_name} (AUC = {roc_auc_lstm:.2f})')
            else:
                # Predict the probabilities of the positive class
                predicted_probabilities = model.predict_proba(X_test)
            
                # Compute ROC curve and AUC
                fpr, tpr, thresholds_roc_test = roc_curve(y_test, predicted_probabilities[:, 1], pos_label=1)
                roc_auc = auc(fpr, tpr)
            
                # Plot ROC curve
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        # Plot the random classifier
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier (AUC = 0.50)')
        
        # Customize the plot
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        
        # Show the plot
        plt.show()


    def plot_confusion_matrix(self, model, model_name, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test):
        '''
        Called by plot_results()
        '''
        
        # Computating the confusion matrix
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_test = confusion_matrix(y_test, y_pred_test)    

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
        
        x_labels = ["Predicted\nNon-Sepsis", "Predicted\nSepsis"]
        y_labels = ["Actual Non-Sepsis", "Actual Sepsis"]
        sns.heatmap(cm_train, annot=True, fmt='d', xticklabels=x_labels, yticklabels=y_labels, ax=axes[0])
        sns.heatmap(cm_test, annot=True, fmt='d', xticklabels=x_labels, yticklabels=y_labels, ax=axes[1])
        
        axes[0].set_title("CM in training set", fontsize = 10)
        axes[1].set_title("CM in test set", fontsize = 10)
        axes[0].tick_params(labelsize=9)
        axes[1].tick_params(labelsize=9)
        plt.tight_layout()
        
        print(model_name)
        print(" ---------------------------------------- ")
        plt.show()


    def plot_precision_recall(self, model, model_name, X_test, y_test):
        # Plot precision recall
        display = PrecisionRecallDisplay.from_estimator(
        model, X_test, y_test, name=model_name, plot_chance_level=True
        )
        display.ax_.legend(loc='upper right')
        _ = display.ax_.set_title("2-class Precision-Recall curve")

        fig = display.figure_
        plt.show()


    def plot_roc_curve(self, predicted_probabilities, model_name, y_test, label = None):
        """_summary_
        returns AUC score, ROC curve, precision recall

        Args:
            fpr_test (array): _description_
            tpr_test (array): _description_
            model_name (str): Model name
            label (_type_, optional): _description_. Defaults to None.
        """
        # Get roc
        fpr, tpr, thresholds_roc_test = roc_curve(y_test, predicted_probabilities[:, 1], pos_label=1)

        # Compute AUC score
        auc_test = auc(fpr, tpr)
        print(f"{model_name} AUC : {auc_test:.4f}")
        
        # Plot ROC curve
        plt.plot(fpr, tpr, linewidth=2, label = label, color='orange')
        plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
        plt.axis([0, 1, 0, 1])                                    
        plt.xlabel('False Positive Rate (Fall-Out)', fontsize=11) 
        plt.ylabel('True Positive Rate (Recall)', fontsize=11)
        plt.grid(False)  
        
        plt.title("ROC Curve: " + model_name)
        plt.figure(figsize=(6, 6))
        plt.show()


    def tune_hyperparameters(self, X_train, y_train, class_weights, candidate_models):
        """    
        Tune hyperparameters for multiple classifiers using GridSearchCV.

        Args:
            X_train (pd.DataFrame or np.array): Feature matrix for training.
            y_train (pd.Series or np.array): Target labels for training.
            class_weights (dict): Class weights for handling class imbalance.
            candidate_models:

        Returns:
            dict: Best parameters for each classifier.
        """  

        # define hyperparameter grid
        param_grid = {
            'Logistic_Regression': {'C': [0.1, 1, 10],
                                    'penalty': ['l2', None]
                                },
            'Random_Forest': {'n_estimators': [50, 100, 150], 
                            'max_depth': [None, 10, 20], 
                            'min_samples_split': [2, 25, 50, 100, 250],
                            'min_samples_leaf': [2, 25, 50, 100, 250]
                            },
            'Gradient_Boosting': {'n_estimators': [50, 100, 150, 200], 
                                'learning_rate': [0.01, 0.1, 0.2, 0.5], 
                                'loss': ['log_loss', 'exponential'], 
                                'criterion': ['friedman_mse', 'squared_error'],
                                'max_features': ['sqrt', 'log2']
                                }
        }
        
        # Create an empty dictionary to store the best parameters for each model
        best_params = {}
        best_models = {}

        # Loop through the candidate_models dictionary to perform GridSearchCV for each model
        for model_name, model in candidate_models.items():
            
            #todo: gridSearchCV or RandomizedSearchCV using refit=True?
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model_name], scoring='balanced_accuracy', cv=3)
            
            if model_name == 'Gradient_Boost':
                grid_search.fit(X_train, y_train, sample_weight=[class_weights[label] for label in y_train])
            else:
                grid_search.fit(X_train, y_train)
            
            
            # Store the best parameters for the current model
            best_params[model_name] = grid_search.best_params_
            best_models[model_name] = grid_search.best_estimator_
                    
            # Get best score
            best_score = grid_search.best_score_
            print(f"Best score for {model_name}: {best_score}")
            print(grid_search.best_params_)
            

            # hypertuned_model = final_model.fit(X_train, y_train)
            
        return best_params
            #return(hypertuned_model)  

    def tune_hyperparameters_rf(self, X_train, y_train, class_weights, model):
        """    
        Tune hyperparameters for Random Forest classifiers using GridSearchCV.

        Args:
            X_train (pd.DataFrame or np.array): Feature matrix for training.
            y_train (pd.Series or np.array): Target labels for training.
            class_weights (dict): Class weights for handling class imbalance.
            candidate_models:

        Returns:
            dict: Best parameters for each classifier.
        """  

        # define hyperparameter grid
        param_grid = {'n_estimators': [50, 100, 150], 
                      'max_depth': [10, 20, 30, 40], 
                      'min_samples_split': [2, 25, 50, 100, 250, 400],
                      'min_samples_leaf': [2, 25, 50, 100, 250, 400]
        }
        
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='balanced_accuracy', cv=3)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

        # Get best score
        print("Best parameters:", best_params)
        print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
            
        return best_params



    def train_models(self, candidate_models, class_weights, X_train, X_test, y_train, y_test):
        
        # Get model performances
        model_performance_df = self.static_models(candidate_models, class_weights, X_train, y_train, X_test, y_test)
        
        # sort by model performance 
        sorted_model_performance_df = model_performance_df.sort_values(by=['average score'],ascending=False) 
        
        # find best performing model
        best_model_name = sorted_model_performance_df.head(1)['model'].to_string().split(" ")[4]
        print("\n\n The best Performing model :", best_model_name)
        
        # tune hyperparameters of best performing model
        tuned_model = self.tune_hyperparameters(X_train, y_train, best_model_name)
            
        # Model performance for test data generated using train test split
        # performance_report = validate_test_groundtruth(tuned_model,X_test_scaled,y_test) 


    def validate_test_groundtruth(self, final_model, X_test_scaled, y_test):
        """
        Model Performance validation against test data

        Args:
            final_model (_type_): _description_
            X_test_scaled (_type_): _description_
            y_test (_type_): _description_
        """
            
        y_predicted = final_model.predict(X_test_scaled)

        print("\n\nFor test data generated using train test split \n")
        print(classification_report(y_test, y_predicted))


        print(f"Cohen_kappa_score: {cohen_kappa_score(y_test, y_predicted)} ")
        
            
    def generate_shap_bar_plot(self, shap_values, model_name, time_point, num_features):
        """
        Generates a bar plot of SHAP values to visualise the global feature importance. Saves it as a PNG file in the output directory with a name incorporating model name, time point, and feature count.

        Args:
            shap_values : The SHAP values computed from the model, indicating feature impacts.
            model_name : str, The name of the model, used for constructing the file name.
            time_point : str, An identifier for the dataset's time point, used in the file name.
            num_features : int, The number of features in X_train, used for informative file naming.
        """
        plt.figure()
        shap.plots.bar(shap_values, show=False)
        plt.savefig(self.output_dir / f"{model_name}_{time_point}_{num_features}_features_bar_plot.png")
        plt.close()

    def generate_shap_summary_plot(self, shap_values, X_test, model_name, time_point, num_features):
        """_summary_

            Generates a SHAP summary plot to show the value distribution of features. Saves the image to the output directory, and then closes the matplotlib plot.
        
        Args:
            shap_values : The SHAP values computed from the model, indicating feature impacts.
            model_name : str, The name of the model, used for constructing the file name.
            time_point : str, An identifier for the dataset's time point, used in the file name.
            num_features : int, The number of features in X_train, used for informative file naming.

        """
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(self.output_dir / f"{model_name}_{time_point}_{num_features}_features_summary_plot.png")
        plt.close()

    def generate_shap_heatmap(self, shap_values, model_name, time_point, num_features):
        """
        Generates a heatmap from SHAP values.

        Args:
            shap_values : SHAP values for the model's output explanation.
            X_test : DataFrame, The test dataset, showing value distribution of features.
            model_name : str, Name of the model, used in file naming.
            time_point : str, Dataset's time identifier, used for file naming.
            num_features : int, Count of features, used in file naming for clarity.
        """
        plt.figure()
        shap.plots.heatmap(shap_values[:2000], show=False)  # Adjust data points limit as needed
        plt.savefig(self.output_dir / f"{model_name}_{time_point}_{num_features}_features_heatmap.png")
        plt.close()

    def generate_and_save_shap_plots_old(self, candidate_models, X_train, X_test, time_point):
        """
            Manages the generation and storage of SHAP plots for various models and datasets.

        Args:
            candidate_models : dict, Models to analyze, keyed by name with model objects as values.
            X_train : DataFrame, Training dataset, used for background distribution in SHAP.
            X_test : DataFrame, Test dataset, upon which SHAP values are calculated.
            time_point : str, Data's time point, used for organizing saved plots.

        Functionality:
            Iterates through each model, determines the correct SHAP explainer, computes SHAP values, and invokes plotting functions to generate and save the SHAP plots. It incorporates the time point and feature count into file names for clarity.
        """
        
        # get number of features
        num_features = X_train.shape[1]
        
        for model_name, model in candidate_models.items():
            # Determine the type of explainer based on the model
            if model_name == 'Logistic_Regression':
                explainer = shap.LinearExplainer(model, X_train)
            else:
                explainer = shap.TreeExplainer(model, X_train)
            
            shap_values = explainer(X_test)

            # Generate and save plots
            self.generate_shap_bar_plot(shap_values, model_name, time_point, num_features)
            self.generate_shap_summary_plot(shap_values, X_test, model_name, time_point, num_features)
            self.generate_shap_heatmap(shap_values, model_name, time_point, num_features)

            print(f"Saved SHAP plots for {model_name} at time {time_point} with {num_features} features in {self.output_dir}")
            
    def generate_and_save_shap_plots(self, candidate_models, X_train, X_test, time_point):
        """
        Generate and save SHAP plots for given models, training data, and test data.

        Parameters:
            models (dict): Dictionary containing model objects.
            X_train (DataFrame): Training data.
            X_test (DataFrame): Testing data.
            time_point (str): The time point label for the data.

        Returns:
            None
        """
        num_features = X_train.shape[1]  # Getting the number of features from X_train

        for model_name, model in candidate_models.items():
            # Check if the model is Logistic Regression
            if model_name == 'Logistic_Regression':
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer(X_test)

                # Save plots, but not heatmap
                self.save_shap_plots(shap_values, X_test, model_name, time_point, num_features, include_heatmap=False)

            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_test)

                # Save all SHAP plots
                self.save_shap_plots(shap_values, X_test, model_name, time_point, num_features, include_heatmap=True)
                        
    def save_shap_plots(self, shap_values, X_test, model_name, time_point, num_features, include_heatmap=True):
        """
        Save SHAP plots to the specified directory.

        Parameters:
            shap_values: The SHAP values for the model.
            X_test (DataFrame): Testing data.
            model_name (str): The name of the model.
            time_point (str): The time point label for the data.
            num_features (int): The number of features in the dataset.
            include_heatmap (bool): Flag to include heatmap in the output.

        Returns:
            None
        """
        output_dir = os.path.join(self.ROOT_DIR, 'output', 'plots', model_name, time_point)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # File name pattern including the number of features
        file_name_pattern = f"{model_name}_{time_point}_{num_features}features"

        # Save bar plot
        plt.figure()
        shap.plots.bar(shap_values)
        plt.savefig(os.path.join(output_dir, f'shap_{file_name_pattern}_summary_bar.png'))

        # Save summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_test)
        plt.savefig(os.path.join(output_dir, f'shap_{file_name_pattern}_summary.png'))

        # Save heatmap if include_heatmap is True
        if include_heatmap:
            plt.figure()
            shap.plots.heatmap(shap_values[:])
            plt.savefig(os.path.join(output_dir, f'shap_{file_name_pattern}_heatmap.png'))

        plt.close('all')  # Close all figures to free memory


    def shap_global_lr(candidate_models, x_test_df, x_train_df):
        explainer_lr = shap.LinearExplainer(candidate_models['Logistic_Regression'], x_train_df)
        shap_values_lr = explainer_lr(x_test_df)

        # set a display version of the data to use for plotting (has string values)
        shap_values_lr.display_data = shap.datasets.adult(display=True)[0].values
        
        shap.plots.bar(shap_values_lr)
        shap.summary_plot(shap_values_lr, x_test_df)
        shap.plots.heatmap(shap_values_lr[:])
    
    def shap_global_rf(candidate_models, x_test_df):
        """
        Args:
            x_test_df (dataframe): 
        """
        # compute SHAP values
        explainer_rf = shap.TreeExplainer(candidate_models['Random_Forest']) 
        shap_values_rf = explainer_rf.shap_values(x_test_df)
        
        # summary
        shap.summary_plot(shap_values_rf, x_test_df)
        
        # summary 2
        shap.summary_plot(shap_values_rf[0], x_test_df)
        
        # heatmap
        shap_values_rf_subset = explainer_rf(x_test_df.iloc[:])
        shap.plots.heatmap(shap_values_rf_subset[:, :, 1])
        
    def shap_global_gb(candidate_models, x_test_df):
        # compute SHAP values
        explainer_gb = shap.TreeExplainer(candidate_models['Gradient_Boosting'])
        shap_values_gb = explainer_gb(x_test_df)
        
        shap.plots.bar(shap_values_gb)
        
        shap.summary_plot(shap_values_gb, x_test_df)
        
        shap.plots.heatmap(shap_values_gb[:])
    
    def plot_combined_roc_curves_lstm(self, models, split_data_dict, top_features, time_windows):
        plt.figure(figsize=(6, 6))
        row_list = []
        # Predict the test set using the best random forest regressor
        for top in top_features:
            for time in time_windows:
                model = models[top][time]
                x_train = split_data_dict[(top, time)]['x_train']
                y_train = split_data_dict[(top, time)]['y_train']
                x_test = split_data_dict[(top, time)]['x_test']
                y_test = split_data_dict[(top, time)]['y_test']
                
                y_train = np.argmax(y_train, axis=1)
                y_test = np.argmax(y_test, axis=1)

                preds_train = model.predict(x_train)
                preds_test = model.predict(x_test)
                # Plotting ROC curve
                fpr, tpr, thresholds_roc_rf_test = roc_curve(y_test, preds_test[:, 1], pos_label=1)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, linewidth=2, label = f'{time} (AUC = {roc_auc:.3f})')

                # Computing f1 and acc
                f1_train = f1_score(y_train, np.argmax(preds_train, axis=1))
                f1_test = f1_score(y_test, np.argmax(preds_test, axis=1))
                acc_train = balanced_accuracy_score(y_train, np.argmax(preds_train, axis=1))
                acc_test = balanced_accuracy_score(y_test, np.argmax(preds_test, axis=1))
                # Computing precision and recall
                precision_train = precision_score(y_train, np.argmax(preds_train, axis=1))
                precision_test = precision_score(y_test, np.argmax(preds_test, axis=1))
                recall_train = recall_score(y_train, np.argmax(preds_train, axis=1))
                recall_test = recall_score(y_test, np.argmax(preds_test, axis=1))
                
                new_row = [time, round(acc_train, 3), round(acc_test, 3),
                            round(precision_train, 3), round(precision_test, 3),
                            round(recall_train, 3), round(recall_test, 3),
                            round(f1_train, 3), round(f1_test, 3),
                            round(roc_auc, 3)]
                row_list.append(new_row)
        
        plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=11)
        plt.ylabel('True Positive Rate', fontsize=11)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.grid(False)
        plt.show()
        table = pd.DataFrame(row_list, columns= ['Model', 'Balanced acc Train', 'Balanced acc Test',
                                                'Precision Train', 'Precision Test',
                                                'Recall Train', 'Recall Test',
                                                'F1 Train', 'F1 Test', 'AUC'])
        return table
    
    def compute_sample_weights(self, y):
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        sample_weights = class_weights[y]
        return sample_weights
    
    def print_results(self, M, X_train, Y_train, X_test, Y_test):
        Y_train = np.argmax(Y_train, axis=1)
        Y_test = np.argmax(Y_test, axis=1)

        preds_train = M.predict(X_train)
        preds_train = np.argmax(preds_train, axis=1)
        conf_mat_train = confusion_matrix(Y_train, preds_train)
        # conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

        preds_test = M.predict(X_test)
        preds_test = np.argmax(preds_test, axis=1)
        conf_mat_test = confusion_matrix(Y_test, preds_test)
        # conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

        print("***[RESULT]*** ACT  Confusion Matrix")
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

        x_labels = ["Predicted\nNon-Sepsis", "Predicted\nSepsis"]
        y_labels = ["Actual Non-Sepsis", "Actual Sepsis"]
        sns.heatmap(conf_mat_train, fmt='d', annot=True, xticklabels=x_labels, yticklabels=y_labels, ax=axes[0])
        sns.heatmap(conf_mat_test, fmt='d', annot=True, xticklabels=x_labels, yticklabels=y_labels, ax=axes[1])

        axes[0].set_title("CM in training set", fontsize = 10)
        axes[1].set_title("CM in test set", fontsize = 10)
        axes[0].tick_params(labelsize=9)
        axes[1].tick_params(labelsize=9)
        plt.tight_layout()
        plt.show()

        f1_train = f1_score(Y_train, preds_train)
        f1_test = f1_score(Y_test, preds_test)
        acc_train = balanced_accuracy_score(Y_train, preds_train)
        acc_test = balanced_accuracy_score(Y_test, preds_test)
        # Computing precision and recall
        precision_train = precision_score(Y_train, preds_train)
        precision_test = precision_score(Y_test, preds_test)
        recall_train = recall_score(Y_train, preds_train)
        recall_test = recall_score(Y_test, preds_test)
        table = pd.DataFrame({'F1-Score':[f1_train, f1_test], 'Balanced Acc':[acc_train, acc_test],
                             'Precision':[precision_train, precision_test], 'Recall':[recall_train, recall_test]},
                             index=['Train', 'Test'])
        return table

    def plot_precision_recall_lstm(self, model, model_name, X_test, y_test):
        # Assuming you have predicted probabilities (change this to your actual predictions)
        y_pred_prob = model.predict(X_test)
        y_pred_prob = y_pred_prob[:, 1]

        # Assuming y_test contains the true labels
        y_true = y_test[:, 1]

        # Calculate the AP value
        ap = round(average_precision_score(y_true, y_pred_prob), 2)

        # Print the AP value
        print(f'Average Precision (AP): {ap:.4f}')

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

        # Create the Precision-Recall curve
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, marker='.', label=f'LSTM (AP = {ap})')
        plt.axhline(y=0.12, color='black', linestyle='--', label='Chance level (AP = 0.12)', xmin=0, xmax=1)
        plt.xlabel('Recall (Positive label: 1)')
        plt.ylabel('Precision (Positive label: 1)')
        plt.legend(loc='upper right')
        plt.title('Precision-Recall Curve')
        plt.grid(False)
        plt.show()