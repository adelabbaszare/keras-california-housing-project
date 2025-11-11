California Housing Price Prediction & Comprehensive EDA
=======================================================

This project develops a deep learning model using **Keras/TensorFlow** to predict median house values in California, based on the built-in `california_housing` dataset. Beyond core model training, it features a complete **Exploratory Data Analysis (EDA)** and visualization component using **Pandas, Matplotlib, and Seaborn**.

🎯 Project Goals
----------------

1.  **Data Analysis:** Perform comprehensive EDA on the housing dataset, including feature distribution, correlation analysis, and geographical plotting.

2.  **Model Development:** Build, train, and evaluate a sequential **Neural Network (NN)** for the regression task.

3.  **Visualization:** Generate static plots to monitor model training history (Loss/MAE) and compare actual vs. predicted prices.

4.  **Interactive Prep:** Prepare and export clean data for powerful interactive visualization using the **SandDance** tool.

💾 Dataset Overview
-------------------

The dataset includes aggregated data for districts across California, focusing on eight key features:

|

Feature Name

 |

Description

 |
|

**MedInc**

 |

Median income for households within a block group.

 |
|

**HouseAge**

 |

Median house age within a block group.

 |
|

**AveRooms**

 |

Average number of rooms per household.

 |
|

**AveBedrms**

 |

Average number of bedrooms per household.

 |
|

**Population**

 |

Block group population.

 |
|

**AveOccup**

 |

Average house occupancy.

 |
|

**Latitude**

 |

Block group latitude (Y-axis).

 |
|

**Longitude**

 |

Block group longitude (X-axis).

 |
|

**MedHouseVal** (Target)

 |

Median house value (in $100,000s).

 |

⚙️ Setup and Installation
-------------------------

### 1\. Python Environment

**IMPORTANT:** It is strongly recommended to use a stable Python version, such as **Python 3.11 or 3.12**, as newer versions (like 3.13) may lack stable TensorFlow binaries.

First, create and activate a virtual environment:

```
# Create environment (using python3.12, adjust if necessary)
python3.12 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
.\venv\Scripts\activate

```

### 2\. Install Dependencies

Install all necessary libraries, including TensorFlow, Pandas, and visualization tools, using the updated `requirements.txt` file.

```
pip install -r requirements.txt

```

🚀 Execution
------------

Run the main training and analysis script from your active virtual environment:

```
python train.py

```

The script will handle the entire workflow, from data loading and EDA visualization to model training and evaluation.

📊 Results and Outputs
----------------------

Upon successful execution, the following outputs will be generated:

### 1\. Static Plots (`plots/`)

A new directory named `plots/` will be created, containing several key visualizations:

-   `1_feature_distributions.png`: Histograms showing the distribution of all input features.

-   `2_correlation_heatmap.png`: Visualizes the correlation matrix between all variables (features and target).

-   `3_geo_price_map.png`: A geographical scatter plot showing house prices across California based on Latitude/Longitude.

-   `4_training_history.png`: **Crucial for Model Monitoring.** Shows the Training and Validation Loss (MSE) and MAE over 100 epochs.

-   `5_predictions_vs_actual.png`: A scatter plot comparing the model's predictions against the actual test values.

### 2\. Exported Data and Model

-   `california_housing_for_sanddance.csv`: A cleaned CSV file containing the training data, exported specifically for interactive visualization tools.

-   `california_housing_model.keras`: The final, trained Keras model, saved for future inference.

🌐 Interactive Visualization with SandDance
-------------------------------------------

The `train.py` script prepares the data but does not run SandDance itself. You can use the exported CSV for powerful, interactive 3D visualizations.

**Steps to Visualize:**

1.  Ensure you have run `train.py` to generate the file: **`california_housing_for_sanddance.csv`**.

2.  Open the **SandDance Viewer** in your browser: [https://sanddance.js.org/app/](https://www.google.com/search?q=https://sanddance.js.org/app/ "null")

3.  Drag and drop the `california_housing_for_sanddance.csv` file onto the page.

4.  Explore the data interactively (e.g., create a 3D plot with **Longitude** on X, **Latitude** on Y, and colored by **MedHouseVal**).
