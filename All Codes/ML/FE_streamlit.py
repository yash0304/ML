import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV,SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import eda
import train as t


st.title("Feature Selection algorithms")
uploaded_file = st.file_uploader("Enter any dataset to upload (in CSV format please!)",type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    feature_df = df.drop(df.columns[-1],axis=1)
    target_df = df[df.columns[-1]]
    
    #st.write("Feature data and target data will automatically be created.")
    # Preview of data
    st.write("Preview of your data")
    st.dataframe(df.head())
    st.write("information about your data")
    summary_df = pd.DataFrame({'columns':df.columns,'Non null counts':df.notnull().sum(),'data type':df.dtypes.astype(str)})
    st.dataframe(summary_df)

    convert_column = st.selectbox("Select column to convert to numeric",options=['--select a column--']+list(df.columns))
    if convert_column!= '--select a column--':
        df[convert_column] = eda.convert_dtype(df[convert_column],'numeric')

    #asking user based target columns
    target_column = st.selectbox("Select target column",options=['--select a column--']+list(df.columns))
    if target_column != '--select a column--':
        st.info(f"column {target_column} has been selected as target")
    
    deletion_columns = st.multiselect("Select any column to delete (Can select multiple)",
                                     options=['--select a column--']+ list(df.columns))

    df_new = df.copy()
    if deletion_columns != '--select a column--':
        df_new = eda.del_cols(df,deletion_columns)
        st.success(f"Deleted Columns ,{''.join(deletion_columns)} ")
    else:
        st.info("You have not selected any column to be deleted")

    remove_null = st.checkbox("If you want to remove null values")

    if remove_null:
        df_new = eda.drop_null_values(df_new)

    method = st.selectbox("Enter your choice of feature selection method",
                 ('--Select a method--','Recursive Feature engineering(RFE)','Sequential Feature Selector','PCA','LDA')
                 ) 


    X = df_new.drop(target_column,axis=1)    
    if target_column == 'Churn':
        df_new[target_column] = df_new[target_column].map({'Yes':1,'No':0})
    y = df_new[target_column]

    cat_cols = eda.get_categorical_cols(X)
    num_cols = eda.get_numerical_cols(X)
    st.write("Numerical Columns")
    st.write(list(num_cols))
    st.write("Categorical Columns")
    st.write(list(cat_cols))
    if cat_cols:
         X_encoded = pd.get_dummies(X[cat_cols],drop_first=True)
         if num_cols:
              X[num_cols] = StandardScaler().fit_transform(X[num_cols])
              #X_num = X[num_cols]
         X = pd.concat([X_encoded,X[num_cols]],axis=1)
    else:
         X[num_cols] = StandardScaler().fit_transform(X[num_cols])
         X = X[num_cols]
    
    st.write("Total columns to go into train test split")
    st.write(X.columns)
    X_train,X_test,y_train,y_test = t.get_train_test_split_data(X,y)
    st.write("X_train columns")
    st.write(list(X_train.columns))
    st.write("Shows null values")
    st.write("X_train null values",X_train.isnull().sum())
    st.write("y_train null values",y_train.isnull().sum())
    
    model = LogisticRegression(penalty='l1',solver = 'liblinear',max_iter=300)

    if method == 'Recursive Feature engineering(RFE)':
        rfe = RFECV(model,min_features_to_select=1,cv=5,scoring='accuracy')
        rfe.fit(X_train,y_train)
        selected_features = X_train.columns[rfe.support_]

        st.subheader("Selected features")
        st.write(list(selected_features))

        X_train_rfe = pd.DataFrame(X_train,columns = X_train.columns[rfe.support_])
        X_test_rfe = pd.DataFrame(X_test,columns=X_test.columns[rfe.support_])
        rfe2 = rfe.fit(X_train_rfe,y_train)
        rfe2_pred = rfe2.predict(X_test_rfe)
        st.write("Accuracy of this model is :   " )
        st.write(f"{accuracy_score(y_test,rfe2_pred)*100} %")

        st.subheader("Accuracy vs. number of feature selectors")
        fig,ax = plt.subplots()
        ax.plot(range(1,len(rfe.cv_results_['mean_test_score'])+1),rfe.cv_results_['mean_test_score'])
        ax.set_title("RFECV feature selection curve")
        ax.set_xlabel("Number of selected features")
        ax.set_ylabel("accuracy")
        st.pyplot(fig)

    elif method =='Sequential Feature Selector':
        num_features = st.slider("Please select features",1,min(X_train.shape[1],50),5)
        sfs = SequentialFeatureSelector(model,n_features_to_select=num_features,direction='forward')
        sfs.fit(X_train,y_train)

        selected_features = X_train.columns[sfs.get_support()]
        st.subheader("Selected features")
        st.write(list(selected_features))

        #transforming into dataframe with slected features so that new accuracy can be looked. 
        X_train_sfs = pd.DataFrame(sfs.transform(X_train),columns = sfs.get_feature_names_out(),index=X_train.index)
        X_test_sfs = pd.DataFrame(sfs.transform(X_test),columns = sfs.get_feature_names_out(),index=X_test.index)
        model.fit(X_train_sfs,y_train)
        y_pred_sfs = model.predict(X_test_sfs)

        st.write("Accuracy of this model is :   " )
        st.write(f"{accuracy_score(y_test,y_pred_sfs)*100} %")

        fig,ax = plt.subplots()
        ax.barh(selected_features,range(len(selected_features)))
        ax.invert_yaxis()
        ax.set_title("Feature ranking")
        ax.set_xlabel("Selection order by ranking")
        st.pyplot(fig)

    elif method == 'PCA':
        num_features = st.slider("Enter number of features to see accuracy on: ",1,X_train.shape[1],5)
        pca = PCA(n_components=num_features)
        X_train_transformed = pca.fit_transform(X_train)
        X_test_transformed = pca.transform(X_test)
        
        if st.checkbox("Whether you want to see the variance?"):
                         st.write(pca.explained_variance_ratio_)
        model.fit(X_train_transformed,y_train)
        y_pred_pca = model.predict(X_test_transformed)
        st.write("Accuracy score is:  ",accuracy_score(y_test,y_pred_pca))

        if st.checkbox("You want to visualize variance explanation by components?"):
             explain_variances_ratio = pca.explained_variance_ratio_
             cumulative_variance = np.cumsum(explain_variances_ratio)

             fig,ax = plt.subplots()
             ax.plot(range(1,len(cumulative_variance)+1),cumulative_variance)
             ax.set_title("Cumulative variance explained by PCA components")
             ax.set_xlabel("PCA components")
             ax.set_ylabel("Cumulative explained variance")
             ax.grid(True)
             st.pyplot(fig)

    elif method == 'LDA':
         lda = LinearDiscriminantAnalysis()
         X_train_lda = lda.fit_transform(X_train,y_train)
         X_test_lda = lda.transform(X_test)
         selected_features = lda.get_feature_names_out()
         st.write("Selected_features")
         st.write(list(selected_features))
         model_lda = LogisticRegression()
         model_lda.fit(X_train_lda,y_train)
         y_pred_lda = model_lda.predict(X_test_lda)
         st.write("Accuracy score is:  ",accuracy_score(y_test,y_pred_lda))
         