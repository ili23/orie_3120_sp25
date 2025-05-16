import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

# Load and preprocess data


def load_and_prepare_data():
    results = pd.read_csv('data/cleaned_results.csv')
    drivers = pd.read_csv('data/cleaned_drivers.csv')
    races = pd.read_csv('data/cleaned_races.csv')
    circuits = pd.read_csv('data/cleaned_circuits.csv')

    results.rename(columns={'grid_position': 'initial_position',
                            'position': 'finishing_position'}, inplace=True)
    results = results.merge(drivers, on='driverId', how='left')
    results = results.merge(races, on='raceId', how='left')
    results = results.merge(circuits, on='circuitId', how='left')
    results = results[results['year'] >=
                      2016].sort_values(by=['year', 'round'])

    results['driver_dob'] = pd.to_datetime(
        results['driver_dob'], errors='coerce')
    results['age'] = results['year'] - results['driver_dob'].dt.year
    results['race_home'] = results['driver_home'] == results['circuit_country']
    results.sort_values(by=['driverId', 'year', 'round'], inplace=True)
    return results


def add_features(df):
    def rolling_feats(group):
        group['last_result'] = group['finishing_position'].shift(1)
        group['last_5_avg'] = group['finishing_position'].shift(
            1).rolling(5).mean()
        return group

    df = df.groupby('driverId', group_keys=False).apply(rolling_feats)
    df['experience'] = df.groupby('driverId').cumcount()
    dnf_ids = list(range(3, 41)) + [54, 62]
    df['dnf_flag'] = df['statusId'].isin(dnf_ids).astype(int)
    df['dnf_running'] = df.groupby(
        'driverId')['dnf_flag'].cumsum().shift(1).fillna(0).astype(int)
    df['last_dnf'] = df.groupby(
        'driverId')['dnf_flag'].shift(1).fillna(0).astype(int)
    df.dropna(inplace=True)
    return df


def prepare_train_test_data(df):
    features = ['age', 'experience', 'dnf_running', 'initial_position',
                'last_5_avg',
                'race_home',
                'circuitId', 'round']

    X = pd.get_dummies(df[features], drop_first=True)
    y = df['finishing_position']

    train_years = df['year'].unique()[:-1]
    test_years = df['year'].unique()[-1:]
    train_df = df[df['year'].isin(train_years)]
    test_df = df[df['year'].isin(test_years)]

    X_train = pd.get_dummies(train_df[features], drop_first=True)
    X_test = pd.get_dummies(test_df[features], drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_df, X_test_df, train_df['finishing_position'], test_df['finishing_position'], test_df


def train_ols_model(X_train, y_train):
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit()
    return model


def evaluate_model(model, X_test, y_test, test_df):
    X_test_const = sm.add_constant(X_test)
    y_pred = model.predict(X_test_const)
    mae = mean_absolute_error(y_test, y_pred)
    print(model.summary())
    print(f"\nUnclipped MAE on Test Set: {mae:.4f}")
    return y_pred, mae


def compute_ranked_mae(y_test, y_pred, test_df):
    grouped = test_df.groupby(['year', 'round'])
    results = []

    for (year, rnd), group in grouped:
        idx = group.index
        actual = y_test.loc[idx]
        pred = pd.Series(y_pred[idx], index=idx)
        race_mae = mean_absolute_error(actual.rank(), pred.rank())
        results.append({'year': year, 'round': rnd, 'ranked_mae': race_mae})

    df = pd.DataFrame(results)
    print("\nRanked MAE by Race:")
    print(df)
    print(f"\nOverall Ranked MAE: {df['ranked_mae'].mean():.4f}")
    return df['ranked_mae'].mean(), df


def calculate_top_n_metrics(y_true, y_pred, group_df, n):
    unranked_errors, ranked_errors, baseline_errors = [], [], []

    for (_, _), group in group_df.groupby(['year', 'round']):
        idx = group.index
        true_pos = y_true.loc[idx]
        pred_pos = pd.Series(y_pred, index=idx)
        base_pos = group['initial_position']

        top_n_idx = true_pos.nsmallest(n).index

        unranked_errors.extend(
            abs(true_pos.loc[top_n_idx] - pred_pos.loc[top_n_idx]))
        actual_ranks = true_pos.loc[top_n_idx].rank()
        predicted_ranks = pred_pos.loc[top_n_idx].rank()
        ranked_errors.extend(abs(actual_ranks - predicted_ranks))
        baseline_errors.extend(
            abs(true_pos.loc[top_n_idx] - base_pos.loc[top_n_idx]))

    return (
        np.mean(unranked_errors) if unranked_errors else np.nan,
        np.mean(ranked_errors) if ranked_errors else np.nan,
        np.mean(baseline_errors) if baseline_errors else np.nan
    )


def plot_top_n_mae(y_test, y_pred, test_df):
    y_pred_clipped = np.clip(y_pred, 1, 20)

    top_n_results = []
    for n in range(1, 21):
        unranked_mae, ranked_mae, baseline_mae = calculate_top_n_metrics(
            y_test, y_pred_clipped, test_df, n
        )
        top_n_results.append({
            'top_n': n,
            'unranked_mae': unranked_mae,
            'ranked_mae': ranked_mae,
            'baseline_mae': baseline_mae
        })

    top_n_results_df = pd.DataFrame(top_n_results)
    print("\nTop-N Results (Actual Top-N Finishers):")
    print(top_n_results_df.round(3))

    plt.figure(figsize=(12, 7))
    plt.plot(top_n_results_df['top_n'], top_n_results_df['unranked_mae'],
             label='Model Unranked MAE', marker='o')
    plt.plot(top_n_results_df['top_n'], top_n_results_df['ranked_mae'],
             label='Model Ranked MAE', marker='s', linestyle='--')
    plt.plot(top_n_results_df['top_n'], top_n_results_df['baseline_mae'],
             label='Baseline MAE', marker='^', linestyle=':')
    plt.xlabel('Top-N Finishers')
    plt.ylabel('MAE')
    plt.title('Top-N MAE Comparison: Model vs Baseline')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('top_n_ranked_vs_unranked.png', dpi=300)
    plt.show()


def test_regression_assumptions(model, X_train, y_train):
    import seaborn as sns
    from statsmodels.graphics.tsaplots import plot_acf

    # Add constant to predictors
    X_train_const = sm.add_constant(X_train)
    predictions = model.predict(X_train_const)
    residuals = y_train - predictions

    # 1. Constant Variance (Homoscedasticity): Residuals vs Fitted
    plt.figure(figsize=(8, 6))
    sns.residplot(x=predictions, y=residuals, lowess=True,
                  line_kws={'color': 'red'})
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted (Check for Constant Variance)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Independence: Autocorrelation plot of residuals
    plt.figure(figsize=(8, 4))
    plot_acf(residuals, lags=40)
    plt.title('Autocorrelation of Residuals (Check for Independence)')
    plt.tight_layout()
    plt.show()

    # 3. Normality: Q-Q plot
    plt.figure(figsize=(6, 6))
    sm.qqplot(residuals, line='s')
    plt.title('Q-Q Plot of Residuals (Check for Normality)')
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main pipeline execution
# ---------------------------
if __name__ == "__main__":
    results_df = load_and_prepare_data()
    results_df = add_features(results_df)
    X_train, X_test, y_train, y_test, test_df = prepare_train_test_data(
        results_df)

    model = train_ols_model(X_train, y_train)
    y_pred, mae = evaluate_model(model, X_test, y_test, test_df)

    overall_ranked_mae, ranked_df = compute_ranked_mae(y_test, y_pred, test_df)

    baseline_pred = test_df['initial_position']
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    print(f"\nBaseline MAE: {baseline_mae:.4f}")
    print(f"Model MAE: {mae:.4f}")
    print(f"Improvement: {(baseline_mae - mae) / baseline_mae:.2%}")

    plot_top_n_mae(y_test, y_pred, test_df)
    test_regression_assumptions(model, X_train, y_train)
