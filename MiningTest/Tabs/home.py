import streamlit as st
from function import load_data1, load_data, normalization
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import joblib
from PIL import Image
import numpy as np

def app():
    button_clicked = False

    st.title("Penguin Species Classification")
    st.text("")
    img =Image.open('lter_penguins.png')
    st.image(img)
    st.set_option("deprecation.showPyplotGlobalUse", False)

    st.markdown("The **:green[Dataset]** that using for **Train Model**")
    if st.button("Another Sample"):
        button_clicked = True
    if button_clicked:
        data = load_data1()
        updated_df = data.sample(10)
        st.dataframe(updated_df)
    else:
        data = load_data1()
        st.dataframe(data.sample(10))
    df = pd.DataFrame(data)
    dataset_link = "https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data"
    st.markdown(f"Link for Dataset: [Click Here]({dataset_link})")
    st.text("")
    
    img1 =Image.open('culmen_depth.png')
    st.image(img1)
    
    st.title("Why we use this dataset?")
    st.text("1. Real-world application")
    st.text("2. Species diversity")
    st.text("3. Rich attribute information")
    st.text("4. Scientific research")

    st.title("Why penguins must be identified?")
    st.text("1. Conservation and Population Monitoring")
    st.text("2. Ecological Studies")
    st.text("3. Behavioral and Reproductive Studies")
    st.text("4. Species-specific Threats and Conservation Measures")

    st.title("Objectives from this classification")
    st.text("1. Species Identification")
    st.text("2. Pattern Recognition and Characterization")
    st.text("3. Population Monitoring and Conservation")
    st.text("4. Research and Innovation")

    st.text("")
    st.text("")
    st.text("")
    st.header("Numerical data distribution chart")
    # chart data ori
    numeric_columns = data.select_dtypes(include="number").columns
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the data for each column
    for column in numeric_columns:
        ax.plot(data[column], label=column)
    # Set the x-axis ticks and labels
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.index)
    # Set the y-axis label
    ax.set_ylabel("Values")
    # Set the chart title
    ax.set_title("Penguin Dataset Chart")
    # Add a legend
    ax.legend()
    # Display the chart
    st.pyplot(fig)

    st.text("")
    st.text("")
    st.markdown(
        "because the distribution of the existing data is still less equal, the data is normalized so that the data distribution becomes more equal"
    )
    st.header("Numerical data distribution chart after normalization")
    # chart normalize
    numeric_columns = df.select_dtypes(include="number").columns
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[numeric_columns])
    fig, ax = plt.subplots()
    for i, column in enumerate(numeric_columns):
        ax.plot(normalized_data[:, i], label=column)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Index")
    ax.set_ylabel("Normalized Values")
    ax.set_title("Normalized Dataset Chart")
    ax.legend()
    st.pyplot(fig)
    
    st.text("")
    st.text("")
    st.header("Data Distribution")
    st.text("Split data with 70% Training and 30% Testing")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text("Species From dataset:")
        # Assuming the column name is 'species'
        species_counts = data['species'].value_counts()
        
        for species, count in species_counts.items():
            st.write(f"{species}: {count}")
    with col2:
        data["island"] = data["island"].replace({"Torgersen": 0, "Biscoe": 1, "Dream": 2})
        data["sex"] = data["sex"].replace({"MALE": 0, "FEMALE": 1})
        data["species"] = data["species"].replace(
            {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
        )
        
        x_data = data[
            [
                "island",
                "culmen_length_mm",
                "culmen_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
                "sex",
            ]
        ]
        y_target = data["species"]

        scaler = MinMaxScaler()
        x_data = scaler.fit_transform(x_data)
        x_data = pd.DataFrame(
            x_data,
            columns=[
                "island",
                "culmen_length_mm",
                "culmen_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
                "sex",
            ],
        )

        X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_target, test_size=0.33, stratify=y_target
        )
        st.text("Training Dataset:")
        training_species_counts = y_train.value_counts()
        for species, count in training_species_counts.items():
            if species == 0:
                species = "Adelie"
            if species == 1:
                species = "Chinstrap"
            if species == 2:
                species = "Gentoo"
            st.write(f"{species}: {count}")
    with col3:
        st.text("Testing Dataset:")
        testing_species_counts = y_test.value_counts()
        for species, count in testing_species_counts.items():
            if species == 0:
                species = "Adelie"
            if species == 1:
                species = "Chinstrap"
            if species == 2:
                species = "Gentoo"
            st.write(f"{species}: {count}")
        
    st.text("")
    st.text("")
    st.header("Data Processing")
    st.markdown(
        "For dataset processing, using a decision tree algorithm and get a testing accuracy of"
    )
    data["island"] = data["island"].replace({"Torgersen": 0, "Biscoe": 1, "Dream": 2})
    data["sex"] = data["sex"].replace({"MALE": 0, "FEMALE": 1})
    data["species"] = data["species"].replace(
        {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
    )

    x_data = data[
        [
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ]
    ]
    y_target = data["species"]

    scaler = MinMaxScaler()
    x_data = scaler.fit_transform(x_data)
    x_data = pd.DataFrame(
        x_data,
        columns=[
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ],
    )

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
    y_pred = treeClass.predict(X_test)
    treeAccuracy = accuracy_score(y_pred, y_test)
    # joblib.dump(treeAccuracy, "model.sav")
    st.text("Accuracy From This Model: " + str(treeAccuracy * 100) + "%")

    st.text("")
    st.text("")
    st.header("Decission Tree Model for this dataset")
    dot_data = tree.export_graphviz(
        decision_tree=treeClass,
        max_depth=5,
        out_file=None,
        filled=True,
        rounded=True,
        feature_names=[
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ],
        class_names=["Species Adelie", "Species Chinstrap", "Species Gentoo"],
    )
    st.graphviz_chart(dot_data)

    species_labels = ["Adelie", "Chinstrap", "Gentoo"]
    st.text("")
    st.text("")
    st.header("Confusion Matrix from testing in this dataset")
    cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    
    # plt.xticks(np.arange(len(species_labels)), species_labels)
    # plt.yticks(np.arange(len(species_labels)), species_labels)
    # labels = y.unique().astype(str)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="horizontal", colorbar=False)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)
    # st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure

    # if st.checkbox("Edit Desision Tree from our Dataset"):
    #     css = """
    #         <style>
    #         .tooltip {
    #             position: absolute;
    #             top: -5px;
    #             left: 0px;
    #             z-index: 5;
    #             margin-top: 0px;
    #             padding-top: 10px;
    #             display: inline-block;
    #             cursor: help;
    #         }

    #         .tooltip .tooltiptext {
    #             visibility: hidden;
    #             width: 200px;
    #             background-color: #f9f9f9;
    #             color: black;
    #             text-align: center;
    #             border-radius: 4px;
    #             padding: 5px;
    #             position: absolute;
    #             z-index: 10;
    #             bottom: -125%;
    #             left: 25px;
    #             transform: translateX(50%);
    #             opacity: 0;
    #             transition: opacity 0.3s;
    #         }

    #         .tooltip:hover .tooltiptext {
    #             visibility: visible;
    #             opacity: 1;
    #         }
    #         </style>
    #         """
    #     st.markdown(css, unsafe_allow_html=True)
        
    #     # st.title("Your Data: ")
    #     # df = pd.DataFrame(dataframe)
    #     # st.dataframe(df)
    #     # st.text("")
        
    #     st.markdown(
    #         '<div class="tooltip">Alpha for Pruning ℹ️<span class="tooltiptext">Pruning is a way to reduces the size of decision trees by removing sections of the tree that are non-critical and redundant to avoid over-fitting.<br><br>Non-negative float, default=0.00</span></div>',
    #         unsafe_allow_html=True,
    #     )
    #     ccp_alpha1 = st.number_input(" ", min_value=0.0)

    #     st.markdown(
    #         '<div class="tooltip">Class Weight ℹ️<span class="tooltiptext">Determine class weights. The None value indicates that all classes have the same weight. Balanced value is one approach to handle class imbalance in a decision tree. The purpose of a balanced score is to ensure that the formation of a decision tree does not tend to favor the dominant majority class.<br><br>{“None”, “balanced”}, default=”None” </span></div>',
    #         unsafe_allow_html=True,
    #     )
    #     class_weight1 = st.selectbox(" ", (None, "balanced"), key="option1")
    #     # class_weight1 = st.selectbox(' ', (None, 0.5, 1, 2), key='option1') # iki punya vincent

    #     st.markdown(
    #         '<div class="tooltip">Criterion ℹ️<span class="tooltiptext">The function to measure the quality of a split.<br><br>{“gini”, “entropy”}, default=”gini”</span></div>',
    #         unsafe_allow_html=True,
    #     )
    #     criterion1 = st.selectbox(" ", ("gini", "entropy"), key="option2")
    #     # criterion1 = st.selectbox(' ', ('entropy', 'gini', 'log_loss'), key='option2') # iki punya vincent

    #     st.markdown(
    #         '<div class="tooltip">Max Depth ℹ️<span class="tooltiptext">The maximum depth of the tree.<br><br>integer, default=None</span></div>',
    #         unsafe_allow_html=True,
    #     )
    #     max_depth1 = st.selectbox(" ", (None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), key="option3")

    #     st.markdown(
    #         '<div class="tooltip">Max Features ℹ️<span class="tooltiptext">The maximum feature that will be used for learning. If you select the number, that makes machine decide the best feature for learning based on number of features you choose. <br><br>integer, default=None</span></div>',
    #         unsafe_allow_html=True,
    #     )
    #     max_features1 = st.selectbox(" ", (None, 1, 2, 3, 4), key="option4")

    #     st.markdown(
    #         '<div class="tooltip">Max Leaf Nodes ℹ️<span class="tooltiptext">The maximum number of leaf nodes to be built in a decision tree. <br><br>integer, default=None</span></div>',
    #         unsafe_allow_html=True,
    #     )
    #     max_leaf_nodes1 = st.selectbox(" ", (None, 1, 2, 3, 4), key="option5")

    #     st.markdown(
    #         '<div class="tooltip">Max Impurity Decrease ℹ️<span class="tooltiptext">Parameters used to limit the separation at each node based on the desired reduction in impurity. If the impurity decrease exceeds the threshold value specified by Max Impurity Decrease, separation will be performed. However, if the decrease in impurity does not reach the threshold value, then the node will be considered as a terminal node (leaf node) and will not be divided further. Higher values ​​will result in a shallower decision tree with fewer splits, while lower values ​​can result in a deeper decision tree with more splits. <br><br>Non-Negative Float, default=0.0</span></div>',
    #         unsafe_allow_html=True,
    #     )
    #     min_impurity_decrease1 = st.selectbox(
    #         " ", (0.0, 0.1, 0.01, 0.2, 0.02), key="option6"
    #     )

    #     st.markdown(
    #         '<div class="tooltip">Min Samples Leaf ℹ️<span class="tooltiptext">Determines the minimum number of samples that must be in a leaf node. If the number of samples is below this, the node will not be subdivided. <br><br>integer, default=1</span></div>',
    #         unsafe_allow_html=True,
    #     )
    #     min_samples_leaf1 = st.selectbox(" ", (1, 2), key="option7")

    #     st.markdown(
    #         '<div class="tooltip">Min Samples Split ℹ️<span class="tooltiptext">Determine the minimum number of samples needed to perform a knot split. If the number of samples is below this number, no separation will be performed. <br><br>integer, default=1</span></div>',
    #         unsafe_allow_html=True,
    #     )
    #     min_samples_split1 = st.selectbox(" ", (2, 3, 4), key="option8")

    #     st.markdown(
    #         '<div class="tooltip">Min Weight Fraction Leaf ℹ️<span class="tooltiptext">Parameter that controls the minimum sample weight fraction required for a node to become a leaf node (terminal node) in a decision tree. By setting a higher value, the decision tree tends to be simpler with fewer divisions and fewer terminal nodes. Conversely, with lower values, the decision tree tends to be more complex with more divisions and more terminal nodes. <br><br>Non-Negative Float, default=0.0</span></div>',
    #         unsafe_allow_html=True,
    #     )
    #     min_weight_fraction_leaf1 = st.selectbox(" ", (0.0, 0.1, 0.2, 0.3, 0.4, 0.5))

    #     st.markdown(
    #         '<div class="tooltip">Splitter ℹ️<span class="tooltiptext">Determine the best separation strategy to use. If using <b>"best"<b>, for all features, the algorithm selects the "best" point to split, then choose the best feature as the final decision. If using <b>"random"<b>, for all features, the algorithm "randomly" selects a point to split, then choose the best feature as the final decision.<br><br>{“best”, “random”}, default="best"</span></div>',
    #         unsafe_allow_html=True,
    #     )
    #     splitter1 = st.selectbox(" ", ("best", "random"), key="option9")
        
    #     # st.write(dataframe)
        
    #     if st.button("Create"):
    #         treeClass = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha1, class_weight=class_weight1, criterion=criterion1,
    #                                               max_depth=max_depth1, max_features=max_features1, max_leaf_nodes=max_leaf_nodes1,
    #                                               min_impurity_decrease=min_impurity_decrease1, min_samples_leaf=min_samples_leaf1, min_samples_split=min_samples_split1,
    #                                               min_weight_fraction_leaf=min_weight_fraction_leaf1, splitter=splitter1)
    #         treeClass.fit(X_train, y_train)
    #         y_pred = treeClass.predict(X_test)
    #         treeAccuracy = accuracy_score(y_pred, y_test)
    #         # joblib.dump(treeAccuracy, "model.sav")
    #         st.text("Accuracy From your edited Model: " + str(treeAccuracy*100) + "%")
            
    #         st.text("")
    #         st.text("")
    #         st.header("Decission Tree Model for this dataset and your Model")
    #         dot_data = tree.export_graphviz(
    #             decision_tree=treeClass, max_depth=5, out_file=None, filled=True, rounded=True,
    #             feature_names=['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'],
    #             class_names=['Species Adelie', 'Species Chinstrap', 'Species Gentoo']
    #         )
    #         st.graphviz_chart(dot_data)
            
    #         st.text("")
    #         st.text("")
    #         st.header("Confusion Matrix from testing in this dataset and your Model")
    #         cm = confusion_matrix(y_test, y_pred)
    #         plt.figure(figsize=(8,6))
    #         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    #         plt.xlabel("Predicted Labels")
    #         plt.ylabel("True Labels")
    #         st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure