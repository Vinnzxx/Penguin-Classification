# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 20:35:53 2023

@author: USER
"""

import streamlit as st
from function import predict
import pandas as pd
from io import StringIO
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle
from function import predict
from function import load_data1, load_data, normalization
import warnings 

def edit(dataframe):
    css = """
        <style>
        .tooltip {
            position: absolute;
            top: -5px;
            left: 0px;
            z-index: 5;
            margin-top: 0px;
            padding-top: 10px;
            display: inline-block;
            cursor: help;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #f9f9f9;
            color: black;
            text-align: center;
            border-radius: 4px;
            padding: 5px;
            position: absolute;
            z-index: 10;
            bottom: -125%;
            left: 25px;
            transform: translateX(50%);
            opacity: 0;
            transition: opacity 0.3s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)
    
    # st.title("Your Data: ")
    # df = pd.DataFrame(dataframe)
    # st.dataframe(df)
    # st.text("")
    
    st.markdown(
        '<div class="tooltip">Alpha for Pruning ℹ️<span class="tooltiptext">Pruning is a way to reduces the size of decision trees by removing sections of the tree that are non-critical and redundant to avoid over-fitting.<br><br>Non-negative float, default=0.00</span></div>',
        unsafe_allow_html=True,
    )
    ccp_alpha1 = st.number_input(" ", min_value=0.0)

    st.markdown(
        '<div class="tooltip">Class Weight ℹ️<span class="tooltiptext">Determine class weights. The None value indicates that all classes have the same weight. Balanced value is one approach to handle class imbalance in a decision tree. The purpose of a balanced score is to ensure that the formation of a decision tree does not tend to favor the dominant majority class.<br><br>{“None”, “balanced”}, default=”None” </span></div>',
        unsafe_allow_html=True,
    )
    class_weight1 = st.selectbox(" ", (None, "balanced"), key="option1")
    # class_weight1 = st.selectbox(' ', (None, 0.5, 1, 2), key='option1') # iki punya vincent

    st.markdown(
        '<div class="tooltip">Criterion ℹ️<span class="tooltiptext">The function to measure the quality of a split.<br><br>{“gini”, “entropy”}, default=”gini”</span></div>',
        unsafe_allow_html=True,
    )
    criterion1 = st.selectbox(" ", ("gini", "entropy"), key="option2")
    # criterion1 = st.selectbox(' ', ('entropy', 'gini', 'log_loss'), key='option2') # iki punya vincent

    st.markdown(
        '<div class="tooltip">Max Depth ℹ️<span class="tooltiptext">The maximum depth of the tree.<br><br>integer, default=None</span></div>',
        unsafe_allow_html=True,
    )
    max_depth1 = st.selectbox(" ", (None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), key="option3")

    st.markdown(
        '<div class="tooltip">Max Features ℹ️<span class="tooltiptext">The maximum feature that will be used for learning. If you select the number, that makes machine decide the best feature for learning based on number of features you choose. <br><br>integer, default=None</span></div>',
        unsafe_allow_html=True,
    )
    max_features1 = st.selectbox(" ", (None, 1, 2, 3, 4), key="option4")

    st.markdown(
        '<div class="tooltip">Max Leaf Nodes ℹ️<span class="tooltiptext">The maximum number of leaf nodes to be built in a decision tree. <br><br>integer, default=None</span></div>',
        unsafe_allow_html=True,
    )
    max_leaf_nodes1 = st.selectbox(" ", (None, 1, 2, 3, 4), key="option5")

    st.markdown(
        '<div class="tooltip">Max Impurity Decrease ℹ️<span class="tooltiptext">Parameters used to limit the separation at each node based on the desired reduction in impurity. If the impurity decrease exceeds the threshold value specified by Max Impurity Decrease, separation will be performed. However, if the decrease in impurity does not reach the threshold value, then the node will be considered as a terminal node (leaf node) and will not be divided further. Higher values ​​will result in a shallower decision tree with fewer splits, while lower values ​​can result in a deeper decision tree with more splits. <br><br>Non-Negative Float, default=0.0</span></div>',
        unsafe_allow_html=True,
    )
    min_impurity_decrease1 = st.selectbox(
        " ", (0.0, 0.1, 0.01, 0.2, 0.02), key="option6"
    )

    st.markdown(
        '<div class="tooltip">Min Samples Leaf ℹ️<span class="tooltiptext">Determines the minimum number of samples that must be in a leaf node. If the number of samples is below this, the node will not be subdivided. <br><br>integer, default=1</span></div>',
        unsafe_allow_html=True,
    )
    min_samples_leaf1 = st.selectbox(" ", (1, 2), key="option7")

    st.markdown(
        '<div class="tooltip">Min Samples Split ℹ️<span class="tooltiptext">Determine the minimum number of samples needed to perform a knot split. If the number of samples is below this number, no separation will be performed. <br><br>integer, default=1</span></div>',
        unsafe_allow_html=True,
    )
    min_samples_split1 = st.selectbox(" ", (2, 3, 4), key="option8")

    st.markdown(
        '<div class="tooltip">Min Weight Fraction Leaf ℹ️<span class="tooltiptext">Parameter that controls the minimum sample weight fraction required for a node to become a leaf node (terminal node) in a decision tree. By setting a higher value, the decision tree tends to be simpler with fewer divisions and fewer terminal nodes. Conversely, with lower values, the decision tree tends to be more complex with more divisions and more terminal nodes. <br><br>Non-Negative Float, default=0.0</span></div>',
        unsafe_allow_html=True,
    )
    min_weight_fraction_leaf1 = st.selectbox(" ", (0.0, 0.1, 0.2, 0.3, 0.4, 0.5))

    st.markdown(
        '<div class="tooltip">Splitter ℹ️<span class="tooltiptext">Determine the best separation strategy to use. If using <b>"best"<b>, for all features, the algorithm selects the "best" point to split, then choose the best feature as the final decision. If using <b>"random"<b>, for all features, the algorithm "randomly" selects a point to split, then choose the best feature as the final decision.<br><br>{“best”, “random”}, default="best"</span></div>',
        unsafe_allow_html=True,
    )
    splitter1 = st.selectbox(" ", ("best", "random"), key="option9")
    
    dataframe['island'] = dataframe['island'].replace({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
    dataframe['sex'] = dataframe['sex'].replace({'MALE': 0, 'FEMALE': 1})
    dataframe['species'] = dataframe['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
            
    x_data = dataframe[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    y_target = dataframe['species']
    
    # st.write(dataframe)
            
    scaler = MinMaxScaler()
    x_data = scaler.fit_transform(x_data)
    x_data = pd.DataFrame(x_data, columns = ['island','culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','sex'])
            
    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_target, test_size=0.33, stratify=y_target
    )
    
    treeClass = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha1, class_weight=class_weight1, criterion=criterion1,
                                          max_depth=max_depth1, max_features=max_features1, max_leaf_nodes=max_leaf_nodes1,
                                          min_impurity_decrease=min_impurity_decrease1, min_samples_leaf=min_samples_leaf1, min_samples_split=min_samples_split1,
                                          min_weight_fraction_leaf=min_weight_fraction_leaf1, splitter=splitter1)
    treeClass.fit(X_train, y_train)
    y_pred = treeClass.predict(X_test)
    treeAccuracy = accuracy_score(y_pred, y_test)
    # joblib.dump(treeAccuracy, "model.sav")
    st.text("Accuracy From This Model: " + str(treeAccuracy*100) + "%")
    
    st.text("")
    st.text("")
    st.header("Decission Tree Model for this dataset")
    dot_data = tree.export_graphviz(
        decision_tree=treeClass, max_depth=5, out_file=None, filled=True, rounded=True,
        feature_names=['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'],
        class_names=['Species Adelie', 'Species Chinstrap', 'Species Gentoo']
    )
    st.graphviz_chart(dot_data)
    
    st.text("")
    st.text("")
    st.header("Confusion Matrix from testing in this dataset")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure
    
def app():
    st.title("Prediction Pages for Penguins")    
    
    option = st.radio("Select Option", ("Predict Species", "Upload File", "Input Form"))
    if option == "Predict Species":
        warnings.filterwarnings('ignore')
        st.text("")
        # pembagian kolom
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_option = st.selectbox("Island: ", ["Torgersen", "Biscoe", "Dream"])
            # selected_option = st.radio("Island: ", ("Torgersen", "Biscoe", "Dream"))
            if selected_option == "Torgersen":
                island = 0.0
            elif selected_option == "Biscoe":
                island = 1.0
            else:
                island = 2.0
            # island = st.text_input('Masukkan Pulau Asal')
        with col1:
            culmen_length_mm = st.number_input('Culmen Length (mm)')
            if culmen_length_mm > 50.0 or culmen_length_mm < 0.0:
                st.warning("Number not Valid, please input again must be lower than 50 and greater than 0")
        with col2:
            culmen_depth_mm = st.number_input('Culmen Depth (mm)')
            if culmen_depth_mm > 25.0 or culmen_depth_mm < 0.0:
                st.warning("Number not Valid, please input again must be lower than 25 and greater than 0")
        with col2:
            flipper_length_mm = st.number_input('Flipper Length (mm)')
            if flipper_length_mm > 250.0 or flipper_length_mm < 0.0:
                st.warning("Number not Valid, please input again must be lower than 250 and greater than 0")
        with col3:
            body_mass_g = st.number_input('Body Mass (g)')
            if body_mass_g > 6000.0 or body_mass_g < 0.0:
                st.warning("Number not Valid, please input again must be lower than 6000 and greater than 0")
        with col3:
            selected_option = st.selectbox("Sex: ", ["Male", "Female"])
            if selected_option == "Male":
                sex = 0.0
            elif selected_option == "Female":
                sex = 1.0
        # island = st.selectbox('Island',["Torgersen", "Biscoe", "Dream"]);
        # culmen_length_mm = st.number_input('Culmen Length', min_value=0.0, step=0.1)
        # culmen_depth_mm = st.number_input('Culmen Depth', min_value=0.0, step=0.1)
        # flipper_length_mm = st.number_input('Flipper Length', min_value=0.0, step=0.1)
        # body_mass_g = st.number_input('Body Mass', min_value=0.0, step=0.1)
        # sex = st.selectbox('Sex',["MALE", "FEMALE"])
        data = {'island': island,
                'culmen_length_mm': culmen_length_mm,
                'culmen_depth_mm': culmen_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex
                }
        fitur = pd.DataFrame(data, index=[0])
        
        data = load_data1()
        # st.dataframe(data)
        data["island"] = data["island"].replace({"Torgersen": 0, "Biscoe": 1, "Dream": 2})
        data["sex"] = data["sex"].replace({"MALE": 0, "FEMALE": 1})
        x_data = data[["island", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]]
        y_target = data["species"]
        X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_target, test_size=0.33, stratify=y_target
        )
        treeClass = tree.DecisionTreeClassifier(
            ccp_alpha=0.0,
            class_weight=None,
            criterion="entropy",
            max_depth=4,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            splitter="best",
        )
        treeClass.fit(X_train, y_train)
        if st.button("Predict"):
            y_pred = treeClass.predict(fitur)
            if y_pred == "Adelie":
                st.success("Species Penguin Adelie")
            if y_pred == "Chinstrap":
                st.success("Species Penguin Chinstrap")
            if y_pred == "Gentoo":
                st.success("Species Penguin Gentoo")
                
            # st.success("Species Penguin: " + y_pred)
            
            # if (Prediction == 1):
            #     st.success("Species Adelie")
            # elif (Prediction == 2):
            #     st.success("Species Chinstrap")
            # elif (Prediction == 3):
            #     st.success("Species Gentoo")
        
    if option == "Input Form":
        col1, col2, col3, col4 = st.columns(4)
        num_rows = st.number_input("Number of Rows", min_value=1, value=1, step=1)
        data = []
            
        for i in range(num_rows):
            st.header(f"Input Penguin (Data {i+1})")
            row = {}
            # with col1:
            row['island'] = st.selectbox('Island',["Torgersen", "Biscoe", "Dream"], key=f"island_{i}")
            # with col2:
            row['culmen_length_mm'] = st.number_input('Culmen Length', min_value=0.0, step=0.1, key=f"culmen_length_{i}", format='%.1f', help='')
            if row['culmen_length_mm'] > 50.0 or row['culmen_length_mm'] < 0.0:
                st.warning("Number not Valid, please input again must be lower than 50 and greater than 0")
                row['culmen_length_mm'] = 0.0
            # with col3:
            row['culmen_depth_mm'] = st.number_input('Culmen Depth', min_value=0.0, step=0.1, key=f"culmen_depth_{i}", format='%.1f', help='')
            # with col4:
            if row['culmen_depth_mm'] > 25.0 or row['culmen_depth_mm'] < 0.0:
                st.warning("Number not Valid, please input again must be lower than 25 and greater than 0")
                row['culmen_depth_mm'] = 0.0
            row['flipper_length_mm'] = st.number_input('Flipper Length', min_value=0.0, step=0.1, key=f"flipper_length_{i}", format='%.1f')
            if row['flipper_length_mm'] > 250.0 or row['flipper_length_mm'] < 0.0:
                st.warning("Number not Valid, please input again must be lower than 250 and greater than 0")
                row['flipper_length_mm'] = 0.0
            # st.empty()
            # with col1:
            row['body_mass_g'] = st.number_input('Body Mass', min_value=0.0, step=0.1, key=f"body_mass_{i}", format='%.1f'  )
            if row['body_mass_g'] > 6000.0 or row['body_mass_g'] < 0.0:
                st.warning("Number not Valid, please input again must be lower than 6000 and greater than 0")
                row['body_mass_g'] = 0.0
            # with col2:
            row['sex'] = st.selectbox('Sex',["MALE", "FEMALE"], key=f"sex_{i}")
            # with col3:
            row['species'] = st.selectbox('Species',["Adelie", "Chinstrap", "Gentoo"], key=f"species_{i}")
            
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            data.append(row)
        
        dataframe = pd.DataFrame(data)
        st.title("Your Data: ")
        df = pd.DataFrame(dataframe)
        st.dataframe(df)
        
        if st.button("predict"):
            dataframe['island'] = dataframe['island'].replace({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
            dataframe['sex'] = dataframe['sex'].replace({'MALE': 0, 'FEMALE': 1})
            dataframe['species'] = dataframe['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
                    
            x_data = dataframe[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
            y_target = dataframe['species']
            
            scaler = MinMaxScaler()
            x_data = scaler.fit_transform(x_data)
            x_data = pd.DataFrame(x_data, columns = ['island','culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','sex'])
            
            treeClass = pickle.load(open('trainmodel.sav', 'rb'))
            pred_data = treeClass.predict(x_data)
            treeAccuracy = accuracy_score(pred_data,y_target)
            st.text("Accuracy from your data: "+str(treeAccuracy*100)+"%")
            
            st.text("")
            st.text("")
            st.header("Decission Tree from your Data")
            dot_data = tree.export_graphviz(
                    decision_tree=treeClass, max_depth=5, out_file=None, filled=True, rounded=True,
                    feature_names=['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'],
                    class_names=['Species Adelie', 'Species Chinstrap', 'Species Gentoo']
            )
            st.graphviz_chart(dot_data)
                
            st.text("")
            st.text("")
            st.header("Confusion Matrix from your data prediction")
            cm = confusion_matrix(pred_data,y_target)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            st.pyplot(plt.gcf())
            
            if st.checkbox("Edit Desision Tree"):
                edit(dataframe)
        
        
    elif option == "Upload File":
        st.markdown("Make sure your Upload **:green[Dataset]** Structure like down **below**")
        if st.button("Another Sample"):
            button_clicked = True
            if button_clicked:
                data = load_data1()
                updated_df = data.sample(5)
                st.dataframe(updated_df)
        else:
            data = load_data1()
            st.dataframe(data.sample(5))
        
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
                # st.write(bytes_data)
            
                # To convert to a string based IO:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                # st.write(stringio)
            
                # To read file as string:
                # string_data = stringio.read()
                # st.write(string_data)
            
                # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(uploaded_file)
            st.title("Your Data: ")
            df = pd.DataFrame(dataframe)
            st.dataframe(df)
            st.text("")
            
            dataframe['island'] = dataframe['island'].replace({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
            dataframe['sex'] = dataframe['sex'].replace({'MALE': 0, 'FEMALE': 1})
            dataframe['species'] = dataframe['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
                    
            x_data = dataframe[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
            y_target = dataframe['species']
            
            # st.write(dataframe)
                    
            scaler = MinMaxScaler()
            x_data = scaler.fit_transform(x_data)
            x_data = pd.DataFrame(x_data, columns = ['island','culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','sex'])
                    
            # scaler = MinMaxScaler()
            # x_data = scaler.fit_transform(x_data)
                
            # X_train, X_test, y_train, y_test = train_test_split(
            #     x_data, y_target, test_size=0.33, random_state=123
            # )
                
            # treeClass = tree.DecisionTreeClassifier()
            # treeClass.fit(X_train, y_train)
            # y_pred = treeClass.predict(X_test)
            # treeAccuracy = accuracy_score(y_pred, y_test)
            # st.text("Accuracy from your data: "+str(treeAccuracy)+"%")
                    
            treeClass = pickle.load(open('trainmodel.sav', 'rb'))
            pred_data = treeClass.predict(x_data)
            treeAccuracy = accuracy_score(pred_data,y_target)
            st.text("Accuracy from your data: "+str(treeAccuracy*100)+"%")
            
            st.text("")
            st.text("")
            st.header("Decission Tree from your Data")
            dot_data = tree.export_graphviz(
                    decision_tree=treeClass, max_depth=5, out_file=None, filled=True, rounded=True,
                    feature_names=['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'],
                    class_names=['Species Adelie', 'Species Chinstrap', 'Species Gentoo']
            )
            st.graphviz_chart(dot_data)
                
            st.text("")
            st.text("")
            st.header("Confusion Matrix from your data prediction")
            cm = confusion_matrix(pred_data,y_target)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure
            
            if st.checkbox("Edit Desision Tree"):
                edit(dataframe)
            
    # if st.checkbox("Edit Desision Tree"):
        # css = """
        #     <style>
        #     .tooltip {
        #         position: absolute;
        #         top: -5px;
        #         left: 0px;
        #         z-index: 5;
        #         margin-top: 0px;
        #         padding-top: 10px;
        #         display: inline-block;
        #         cursor: help;
        #     }

        #     .tooltip .tooltiptext {
        #         visibility: hidden;
        #         width: 200px;
        #         background-color: #f9f9f9;
        #         color: black;
        #         text-align: center;
        #         border-radius: 4px;
        #         padding: 5px;
        #         position: absolute;
        #         z-index: 10;
        #         bottom: -125%;
        #         left: 25px;
        #         transform: translateX(50%);
        #         opacity: 0;
        #         transition: opacity 0.3s;
        #     }

        #     .tooltip:hover .tooltiptext {
        #         visibility: visible;
        #         opacity: 1;
        #     }
        #     </style>
        #     """
        # st.markdown(css, unsafe_allow_html=True)
        
        # # st.title("Your Data: ")
        # # df = pd.DataFrame(dataframe)
        # # st.dataframe(df)
        # # st.text("")
        
        # st.markdown(
        #     '<div class="tooltip">Alpha for Pruning ℹ️<span class="tooltiptext">Pruning is a way to reduces the size of decision trees by removing sections of the tree that are non-critical and redundant to avoid over-fitting.<br><br>Non-negative float, default=0.00</span></div>',
        #     unsafe_allow_html=True,
        # )
        # ccp_alpha1 = st.number_input(" ", min_value=0.0)

        # st.markdown(
        #     '<div class="tooltip">Class Weight ℹ️<span class="tooltiptext">Determine class weights. The None value indicates that all classes have the same weight. Balanced value is one approach to handle class imbalance in a decision tree. The purpose of a balanced score is to ensure that the formation of a decision tree does not tend to favor the dominant majority class.<br><br>{“None”, “balanced”}, default=”None” </span></div>',
        #     unsafe_allow_html=True,
        # )
        # class_weight1 = st.selectbox(" ", (None, "balanced"), key="option1")
        # # class_weight1 = st.selectbox(' ', (None, 0.5, 1, 2), key='option1') # iki punya vincent

        # st.markdown(
        #     '<div class="tooltip">Criterion ℹ️<span class="tooltiptext">The function to measure the quality of a split.<br><br>{“gini”, “entropy”}, default=”gini”</span></div>',
        #     unsafe_allow_html=True,
        # )
        # criterion1 = st.selectbox(" ", ("gini", "entropy"), key="option2")
        # # criterion1 = st.selectbox(' ', ('entropy', 'gini', 'log_loss'), key='option2') # iki punya vincent

        # st.markdown(
        #     '<div class="tooltip">Max Depth ℹ️<span class="tooltiptext">The maximum depth of the tree.<br><br>integer, default=None</span></div>',
        #     unsafe_allow_html=True,
        # )
        # max_depth1 = st.selectbox(" ", (None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), key="option3")

        # st.markdown(
        #     '<div class="tooltip">Max Features ℹ️<span class="tooltiptext">The maximum feature that will be used for learning. If you select the number, that makes machine decide the best feature for learning based on number of features you choose. <br><br>integer, default=None</span></div>',
        #     unsafe_allow_html=True,
        # )
        # max_features1 = st.selectbox(" ", (None, 1, 2, 3, 4), key="option4")

        # st.markdown(
        #     '<div class="tooltip">Max Leaf Nodes ℹ️<span class="tooltiptext">The maximum number of leaf nodes to be built in a decision tree. <br><br>integer, default=None</span></div>',
        #     unsafe_allow_html=True,
        # )
        # max_leaf_nodes1 = st.selectbox(" ", (None, 1, 2, 3, 4), key="option5")

        # st.markdown(
        #     '<div class="tooltip">Max Impurity Decrease ℹ️<span class="tooltiptext">Parameters used to limit the separation at each node based on the desired reduction in impurity. If the impurity decrease exceeds the threshold value specified by Max Impurity Decrease, separation will be performed. However, if the decrease in impurity does not reach the threshold value, then the node will be considered as a terminal node (leaf node) and will not be divided further. Higher values ​​will result in a shallower decision tree with fewer splits, while lower values ​​can result in a deeper decision tree with more splits. <br><br>Non-Negative Float, default=0.0</span></div>',
        #     unsafe_allow_html=True,
        # )
        # min_impurity_decrease1 = st.selectbox(
        #     " ", (0.0, 0.1, 0.01, 0.2, 0.02), key="option6"
        # )

        # st.markdown(
        #     '<div class="tooltip">Min Samples Leaf ℹ️<span class="tooltiptext">Determines the minimum number of samples that must be in a leaf node. If the number of samples is below this, the node will not be subdivided. <br><br>integer, default=1</span></div>',
        #     unsafe_allow_html=True,
        # )
        # min_samples_leaf1 = st.selectbox(" ", (1, 2), key="option7")

        # st.markdown(
        #     '<div class="tooltip">Min Samples Split ℹ️<span class="tooltiptext">Determine the minimum number of samples needed to perform a knot split. If the number of samples is below this number, no separation will be performed. <br><br>integer, default=1</span></div>',
        #     unsafe_allow_html=True,
        # )
        # min_samples_split1 = st.selectbox(" ", (2, 3, 4), key="option8")

        # st.markdown(
        #     '<div class="tooltip">Min Weight Fraction Leaf ℹ️<span class="tooltiptext">Parameter that controls the minimum sample weight fraction required for a node to become a leaf node (terminal node) in a decision tree. By setting a higher value, the decision tree tends to be simpler with fewer divisions and fewer terminal nodes. Conversely, with lower values, the decision tree tends to be more complex with more divisions and more terminal nodes. <br><br>Non-Negative Float, default=0.0</span></div>',
        #     unsafe_allow_html=True,
        # )
        # min_weight_fraction_leaf1 = st.selectbox(" ", (0.0, 0.1, 0.2, 0.3, 0.4, 0.5))

        # st.markdown(
        #     '<div class="tooltip">Splitter ℹ️<span class="tooltiptext">Determine the best separation strategy to use. If using <b>"best"<b>, for all features, the algorithm selects the "best" point to split, then choose the best feature as the final decision. If using <b>"random"<b>, for all features, the algorithm "randomly" selects a point to split, then choose the best feature as the final decision.<br><br>{“best”, “random”}, default="best"</span></div>',
        #     unsafe_allow_html=True,
        # )
        # splitter1 = st.selectbox(" ", ("best", "random"), key="option9")
        
        # dataframe['island'] = dataframe['island'].replace({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
        # dataframe['sex'] = dataframe['sex'].replace({'MALE': 0, 'FEMALE': 1})
        # dataframe['species'] = dataframe['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
                
        # x_data = dataframe[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
        # y_target = dataframe['species']
        
        # # st.write(dataframe)
                
        # scaler = MinMaxScaler()
        # x_data = scaler.fit_transform(x_data)
        # x_data = pd.DataFrame(x_data, columns = ['island','culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','sex'])
                
        # X_train, X_test, y_train, y_test = train_test_split(
        #     x_data, y_target, test_size=0.33, stratify=y_target
        # )
        
        # treeClass = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha1, class_weight=class_weight1, criterion=criterion1,
        #                                       max_depth=max_depth1, max_features=max_features1, max_leaf_nodes=max_leaf_nodes1,
        #                                       min_impurity_decrease=min_impurity_decrease1, min_samples_leaf=min_samples_leaf1, min_samples_split=min_samples_split1,
        #                                       min_weight_fraction_leaf=min_weight_fraction_leaf1, splitter=splitter1)
        # treeClass.fit(X_train, y_train)
        # y_pred = treeClass.predict(X_test)
        # treeAccuracy = accuracy_score(y_pred, y_test)
        # # joblib.dump(treeAccuracy, "model.sav")
        # st.text("Accuracy From This Model: " + str(treeAccuracy*100) + "%")
        
        # st.text("")
        # st.text("")
        # st.header("Decission Tree Model for this dataset")
        # dot_data = tree.export_graphviz(
        #     decision_tree=treeClass, max_depth=5, out_file=None, filled=True, rounded=True,
        #     feature_names=['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'],
        #     class_names=['Species Adelie', 'Species Chinstrap', 'Species Gentoo']
        # )
        # st.graphviz_chart(dot_data)
        
        # st.text("")
        # st.text("")
        # st.header("Confusion Matrix from testing in this dataset")
        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(8,6))
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        # plt.xlabel("Predicted Labels")
        # plt.ylabel("True Labels")
        # st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure