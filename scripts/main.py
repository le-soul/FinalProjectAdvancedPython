"""
Machine Learning Project's script on Madrid housing prices
"""
import os
import sys
sys.path.append('scripts')
import click
import pandas as pd
import cleaning as cl

from graphs import ViewClass
from mainfunctions import MainClass
from decisiontree import DecisionClass
from linearregression import PredictClass


def load_dataset(input):
    """
    Function to load dataset and raise error if its not .csv
    """

    extension = input.rsplit(".", 1)[-1]
    if extension == "csv":
        return pd.read_csv(input)
    raise TypeError(f"The extension is {extension} and not csv. Try again")


@click.group(help="Test different levels of click")
def cli():
    pass


@cli.command(short_help="Choose which graph to display")
@click.option("-gr", "--graphs", 
              help="Choose which graph to display: correlation, price skewness, most exp districts, most rooms districts, most bathrooms districs, price x variables",)
@click.option("-i", "--input", required=True, help="File to import")
@click.option("-o", "--output", default="outputs", help="Output directory to save the graphs")
@click.option("-dp", "--duplicates", is_flag=True, help="Shows you the duplicates and nulls")
@click.option("-pl", "--plots", is_flag=True, help="Choose to access the options of graphs")
def view_data(graphs, input, output, duplicates, plots):
    """
    Function to view the data
    """
    load = load_dataset(input)
    cleaner = cl.CleaningClass(load)

    if duplicates:
        cleaner.display_null_and_duplicates_info()

    df = cleaner.clean_data()
    viewer = ViewClass(df)

    if not os.path.exists(output):
        os.makedirs(output)

    MainClass.plotter(plots, viewer, graphs, output)


@cli.command(short_help="")
@click.option("-r", "--regression", is_flag=True, help="")
@click.option(
    "-ln",
    "--linearregression",
    help="Choose which to display: regression, multicollinearity+, predict your buying price",
)
@click.option("-i", "--input", required=True, help="File to import")
@click.option(
    "-o", "--output", default="outputs", help="Output directory to save the graphs"
)
@click.option("-css", "--classifier", is_flag=True, help="")
@click.option(
    "-d", "--decisiontree", help="Choose which to display: has ac or has parking"
)
def training(input, output, regression, linearregression, classifier, decisiontree):
    """
    Function to predict buying price based on other dependent variables
    """
    load = load_dataset(input)
    cleaner = cl.CleaningClass(load)
    df = cleaner.clean_data()
    trainer = PredictClass(df)

    MainClass.regression(regression, linearregression, trainer, output)

    tree = DecisionClass(df)
    MainClass.classifier(classifier, decisiontree, tree, output)


if __name__ == "__main__":
    cli()
