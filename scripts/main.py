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

@cli.command(short_help='')
@click.option('-gr', '--graphs', help='To pass on to choose which graph to use')
@click.option("-i", "--input", required=True, help="File to import")

def view_data(graphs, input):
    """
    Function to view the data
    """
    df = cl.CleaningClass.clean_data(load_dataset(input))
    if graphs:
        if graphs ==



    