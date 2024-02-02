"""
Script used to call the functions that made up the options of the click
"""

import os
import sys
sys.path.append('scripts')
import pandas as pd
import matplotlib.pyplot as plt

class MainClass:
    def plotter(plots, viewer, graphs, output):
        """
        Allows to call graphs by choosing the respective graph name
        """

        if plots:
            if graphs == 'correlation':
                graph_path = os.path.join(output, 'correlation_matrix.png')
                viewer.correlation_matrix()
            elif graphs == 'price skewness':
                graph_path = os.path.join(output, 'price_skewness.png')
                viewer.price_skewness()
            elif graphs == 'most exp districts':
                graph_path = os.path.join(output, 'most_exp_districts.png')
                viewer.most_exp_districts()
            elif graphs == 'most rooms districts':
                graph_path = os.path.join(output, 'most_rooms_districts.png')
                viewer.most_rooms_districts()
            elif graphs == 'most bathrooms districts':
                graph_path = os.path.join(output, 'most_bathrooms_districts.png')
                viewer.most_bathrooms_districts()
            elif graphs == 'price x variables':
                graph_path = os.path.join(output, 'scatter_price_x_variables.png')
                viewer.price_and()
            else:
                print("Invalid option for graph. Choose from: correlation, price skewness, most exp districts, most rooms districts, most bathrooms districs, price x variables")
                return
            plt.savefig(graph_path)
            print(f"Graph saved at: {graph_path}") 
    
    def regression(regression, linearregression, trainer, output):
        """
        Allows to call a regression graph, multicollinearity test and more information or be able to predict your buying price by choosing
        """

        if regression:
            if linearregression == "regression":
                graph_path = os.path.join(output, 'Linear_Regression.png')
                trainer.price_as_y()
                plt.savefig(graph_path)
                print(f"Graph saved at: {graph_path}")
            elif linearregression == "multicollinearity+":
                trainer.multicollinearity_and_model_equation()
            elif linearregression == "predict your buying price":
                trainer.predict_price()
            else:
                print("Invalid option for regression. Choose from: regression, multicollinearity+, predict your buying price")
                return
            
    def classifier(classifier, decisiontree, tree, output):
        """
        Allows to call a decision tree based on if it has ac or if it has parking
        """

        if classifier:
            if decisiontree == "has ac":
                graph_path = os.path.join(output, 'Decision_Tree_Ac.png')
                tree.decision_tree_ac()

            elif decisiontree == "has parking":
                graph_path = os.path.join(output, 'Decision_Tree_Parking.png')
                tree.decision_tree_parking()
            plt.savefig(graph_path)
            print(f"Graph saved at: {graph_path}")