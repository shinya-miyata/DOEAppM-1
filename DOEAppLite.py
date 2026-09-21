import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="軽量DOEアプリ", layout="wide")
st.title("🔬 軽量版 実験計画支援アプリ（拡張版）")

# =========================================================
# ① 実験候補生成
# =========================================================
st.header("① 実験候補の生成")

uploaded_csv = st.file_uploader("生成条件CSV（1行目:項目名、2行目:lower、3行目:upper）", type="csv")
sample_size = st.number_input("生成するサンプル数", min_value=10, max_value=5000, value=100)

if uploaded_csv:
    df_bounds = pd.read_csv(uploaded_csv, index_col=0)
    df_bounds = df_bounds.dropna(axis=1, how="all")
    st.write("読み込んだ生成条件:", df_bounds)

    samples = []
    for _ in range(sample_size):
        row = []
        for col in df_bounds.columns:
            lower = float(df_bounds.loc["lower", col])
            upper = float(df_bounds.loc["upper", col])
            row.append(np.random.uniform(lower, upper))
        samples.append(row)

    df_samples = pd.DataFrame(samples, columns=df_bounds.columns)
    st.dataframe(df_samples)

    st.download_button(
        "📥 生成サンプルCSVをダウンロード",
        df_samples.to_csv(index=False),
        file_name="generated_samples.csv"
    )

# =========================================================
# ② D最適基準で候補選択
# =========================================================
st.header("② 実験候補選択（D最適基準）")

uploaded_samples = st.file_uploader("候補データCSV（説明変数のみ）", type="csv")
select_size = st.number_input("選択するサンプル数", min_value=2, max_value=500, value=10)

if uploaded_samples:
    df_all = pd.read_csv(uploaded_samples)
    st.write("候補データ:", df_all)

    best_score = -np.inf
    best_subset = None

    for _ in range(200):  # 軽量ランダム探索
        subset = df_all.sample(n=select_size)
        X = StandardScaler().fit_transform(subset.values)
        score = np.linalg.det(X.T @ X)
        if score > best_score:
            best_score = score
            best_subset = subset

    df_unselected = df_all.drop(best_subset.index)

    st.subheader("選択された候補")
    st.dataframe(best_subset)

    st.subheader("未選択候補")
    st.dataframe(df_unselected)

    st.download_button(
        "📥 選択候補CSVをダウンロード",
        best_subset.to_csv(index=False),
        file_name="selected_dopt.csv"
    )

    st.download_button(
        "📥 未選択候補CSVをダウンロード",
        df_unselected.to_csv(index=False),
        file_name="unselected_dopt.csv"
    )

# =========================================================
# ③ 次の候補選定（複数モデル対応）
# =========================================================
st.header("③ 次の実験候補選定（複数モデル対応）")

uploaded_train = st.file_uploader("実験済みデータ（説明変数＋目的変数）CSV", type="csv")
uploaded_candidates = st.file_uploader("未実験候補（説明変数のみ）CSV", type="csv")

if uploaded_train and uploaded_candidates:
    df_train = pd.read_csv(uploaded_train)
    df_candidates = pd.read_csv(uploaded_candidates)

    target_col = st.selectbox("目的変数を選択", df_train.columns)
    goal_value = st.number_input("目標値", value=float(df_train[target_col].median()))

    model_name = st.selectbox(
        "使用する回帰モデルを選択",
        ["LinearRegression", "PolynomialRegression", "RandomForest"]
    )

    # ===== モデル構築 =====
    def build_model(name):
        if name == "LinearRegression":
            return LinearRegression()
        elif name == "PolynomialRegression":
            return Pipeline([
                ("poly", PolynomialFeatures(degree=2)),
                ("linear", LinearRegression())
            ])
        elif name == "RandomForest":
            return RandomForestRegressor(n_estimators=200, random_state=0)

    model = build_model(model_name)

    # ===== 説明変数の列を揃える =====
    feature_cols = [col for col in df_train.columns if col != target_col]

    missing_cols = [col for col in feature_cols if col not in df_candidates.columns]
    if missing_cols:
        st.error(f"候補データに不足している列があります: {missing_cols}")
    else:
        # ===== データ準備 =====
        X = df_train[feature_cols].values
        y = df_train[target_col].values
        X_pred = df_candidates[feature_cols].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_pred_scaled = scaler.transform(X_pred)

        # ===== モデル学習 =====
        model.fit(X_scaled, y)

        # ===== モデル性能評価 =====
        y_pred_train = model.predict(X_scaled)
        r2 = r2_score(y, y_pred_train)
        rmse = np.sqrt(mean_squared_error(y, y_pred_train))

        st.subheader("📊 モデルの正確性")
        st.write(f"R²（決定係数）: **{r2:.3f}**")
        st.write(f"RMSE（誤差）: **{rmse:.3f}**")

        # ===== 未実験候補の予測 =====
        y_pred = model.predict(X_pred_scaled)

        df_result = df_candidates.copy()
        df_result["Predicted"] = y_pred
        df_result["Distance"] = np.abs(y_pred - goal_value)

        df_sorted = df_result.sort_values("Distance")
        st.subheader("📌 次の候補（目標値に近い順）")
        st.dataframe(df_sorted)

        st.download_button(
            "📥 次候補CSVをダウンロード",
            df_sorted.to_csv(index=False),
            file_name="next_candidates.csv"
        )
