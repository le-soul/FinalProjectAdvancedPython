"""
Machine Learning Project's script on Madrid housing prices
"""
import os
import sys
sys.path.append('scripts')
import click
import pandas as pd
import matplotlib.pyplot as plt
import cleaning as cl

from graphs import ViewClass
from linearregression import PredictClass
from decisiontree import DecisionClass

def load_dataset(input):
    """
    Function to load dataset and raise error if its not .csv
    """

    extension = input.rsplit('.', 1)[-1]
    if extension == "csv":
        return pd.read_csv(input)
    raise TypeError(f"The extension is {extension} and not csv. Try again")

@click.group(help="Test different levels of click")
def cli():
    pass

@cli.command(short_help='Choose which graph to display')
@click.option('-gr', '--graphs', help='Choose which graph to display: correlation, price skewness, most exp districts, most rooms districts, most bathrooms districs, price x variables')
@click.option("-i", "--input", required=True, help="File to import")
@click.option("-o", "--output", default="graphs", help="Output directory to save the graphs")

def view_data(graphs, input, output):
    """
    Function to view the data
    """
    load = load_dataset(input)
    cleaner = cl.CleaningClass(load)
    df = cleaner.clean_data()
    viewer = ViewClass(df)

    if not os.path.exists(output):
        os.makedirs(output)
    if graphs:
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

@cli.command(short_help='')
@click.option('-r', '--regression', is_flag=True, help='')
@click.option('-ln', '--linearregression', help='Choose which to display: regression or multicollinearity+')
@click.option("-i", "--input", required=True, help="File to import")
@click.option("-o", "--output", default="outputs", help="Output directory to save the graphs")
@click.option("-css", "--classifier", is_flag=True, help="")
@click.option("-d", "--decisiontree", help="Choose which to display: has ac or has parking")

def training(input, output, regression, linearregression, classifier, decisiontree):
    """
    Function to predict
    """
    load = load_dataset(input)
    cleaner = cl.CleaningClass(load)
    df = cleaner.clean_data()
    trainer = PredictClass(df)
    if regression:
        if linearregression == "regression":
            graph_path = os.path.join(output, 'Linear_Regression.png')
            trainer.price_as_y()
            plt.savefig(graph_path)
            print(f"Graph saved at: {graph_path}")
        elif linearregression == "multicollinearity+":
            trainer.multicollinearity_and_model_equation()
        else:
            print("Invalid option for regression. Choose from: regression or multicollinearity+")
            return
        
    
    tree = DecisionClass(df)
    if classifier:
        if decisiontree == "has ac":
            graph_path = os.path.join(output, 'Decision_Tree_Ac.png')
            tree.decision_tree_ac()

        elif decisiontree == "has parking":
            graph_path = os.path.join(output, 'Decision_Tree_Parking.png')
            tree.decision_tree_parking()
        plt.savefig(graph_path)
        print(f"Graph saved at: {graph_path}")

if __name__ == "__main__":
    cli()

    