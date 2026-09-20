# GMRモデル モデルの性能をk-foldクロスバリデーションに変更
import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
import optuna
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR, OneClassSVM
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from plotly.figure_factory import create_scatterplotmatrix
from scipy.stats import norm
from scipy.optimize import minimize
from gmr.gmm import GMM

plt.rcParams["font.family"] = "Meiryo"

st.set_page_config(page_title="実験計画アプリ", layout="wide")
st.title("🔬 実験計画支援アプリ")

# 共通関数：CSVダウンロードリンク生成
def generate_download_link(df, filename):
    # 拡張子が .csv で終わっていない場合は補完
    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 サンプルCSVをダウンロード</a>'
    return href

# 変数名を安全な形式に変換
def sanitize_variable_name(name):
    return name.strip().replace(" ", "_").replace("-", "_")

# タブ構成
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["① 実験候補生成", "② 実験候補選択", "③ 回帰モデルによる次の候補選択", "④ ベイズ最適化(候補あり）", "⑤ ベイズ最適化(候補なし）", "⑥ GMR逆解析"])

# ① サンプル生成
with tab1:
    st.header("① 実験候補の生成")
    uploaded_csv = st.file_uploader("📤 生成条件CSVをアップロード（1行目に項目、2行目に最小Lower、3行目に最大値upperの表（csv））", type="csv")
    sample_size = st.number_input("🔢 生成するサンプル数", min_value=10, max_value=10000, value=100)

    if uploaded_csv:
        df_bounds = pd.read_csv(uploaded_csv, index_col=0)
        df_bounds = df_bounds.dropna(axis=1, how='all')
        st.write("📊 読み込んだ生成条件：", df_bounds)

        selected_features = st.multiselect("🔍 制約をかける特徴量を選択　例　time＜=10", df_bounds.columns.tolist())

        st.subheader("🧮 特徴量ごとの範囲指定")
        feature_constraints = {}
        if "constraint_history" not in st.session_state:
            st.session_state["constraint_history"] = {}

        # 特徴量を2分割して左右に表示
        half = len(selected_features) // 2 + len(selected_features) % 2
        left_features = selected_features[:half]
        right_features = selected_features[half:]
        col_left, col_right = st.columns(2)

        def render_constraint_column(features, column):
            with column:
                for feature in features:
                    safe_feature = sanitize_variable_name(feature)
                    with st.expander(f"🔧 {feature} の制約設定", expanded=False):
                        enabled = st.checkbox(f"{feature} の制約を有効にする", value=True, key=f"enable_{feature}")
                        mode = st.radio("制約の種類", ["単一式", "範囲指定"], key=f"mode_{feature}")

                        if mode == "単一式":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                op = st.selectbox("演算子", ["<=", ">=", "=="], key=f"op_{feature}")
                            with col2:
                                val = st.number_input("値", value=10.0, key=f"val_{feature}")
                            with col3:
                                coef = st.number_input("係数", value=1.0, key=f"coef_{feature}")
                            expr = f"{coef}*x {op} {val}"
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                min_val = st.number_input("下限値", value=0.0, key=f"min_{feature}")
                            with col2:
                                max_val = st.number_input("上限値", value=10.0, key=f"max_{feature}")
                            expr = f"{min_val} <= x <= {max_val}"

                        st.code(f"制約式: {expr}")

                        if st.button(f"💾 {feature} の制約式を履歴に保存", key=f"save_{feature}"):
                            st.session_state["constraint_history"].setdefault(feature, []).append(expr)

                        if feature in st.session_state["constraint_history"]:
                            st.markdown("📜 過去の制約式:")
                            for i, hist in enumerate(st.session_state["constraint_history"][feature]):
                                st.markdown(f"- {hist}")

                        if enabled:
                            feature_constraints[safe_feature] = (expr, feature)

        render_constraint_column(left_features, col_left)
        render_constraint_column(right_features, col_right)

        # ✅ 合計制約はループの外に配置
        with st.expander("➕ 合計制約の設定", expanded=False):
            enable_sum_constraint = st.checkbox("合計制約を有効にする", value=True)
            sum_features = st.multiselect("合計に含める特徴量を選択", df_bounds.columns.tolist(), key="sum_features")
            sum_terms = []
            symbol_map = {}

            for feat in sum_features:
                safe_feat = sanitize_variable_name(feat)
                coef = st.number_input(f"{feat} の係数", value=1.0, key=f"sum_coef_{feat}")
                sum_terms.append(f"{coef}*{safe_feat}")
                symbol_map[safe_feat] = feat

            col_op, col_rhs = st.columns([1, 2])
            with col_op:
                sum_op = st.selectbox("演算子", ["<=", ">=", "=="], key="sum_op")
            with col_rhs:
                sum_rhs = st.number_input("合計の閾値", value=10.0, key="sum_rhs")

            sum_constraint_expr = f"{' + '.join(sum_terms)} {sum_op} {sum_rhs}"
            st.code(f"合計制約式: {sum_constraint_expr}")

            if "sum_history" not in st.session_state:
                st.session_state["sum_history"] = []
            if st.button("💾 合計制約式を履歴に保存"):
                st.session_state["sum_history"].append(sum_constraint_expr)

            if st.session_state["sum_history"]:
                st.markdown("📜 過去の合計制約式:")
                for i, hist in enumerate(st.session_state["sum_history"]):
                    st.markdown(f"- {hist}")

        # ✅ 制約チェック関数（範囲式対応）
        def is_valid_sample(sample_dict, feature_constraints, sum_constraint_expr, symbol_map, enable_sum_constraint):
            for safe_feature, (expr, original_feature) in feature_constraints.items():
                x = sp.Symbol("x")
                try:
                    parsed = sp.sympify(expr)
                    constraint_func = sp.lambdify(x, parsed, "numpy")
                    if not constraint_func(sample_dict[original_feature]):
                        return False
                except Exception as e:
                    st.warning(f"制約式の解析に失敗しました（{original_feature}）: {e}")
                    return False

            if enable_sum_constraint and symbol_map:
                try:
                    expr = sp.sympify(sum_constraint_expr)
                    subs = {sp.Symbol(safe): sample_dict[original] for safe, original in symbol_map.items()}
                    if not expr.subs(subs):
                        return False
                except Exception as e:
                    st.warning(f"合計制約式の解析に失敗しました: {e}")
                    return False

            return True

        # ✅ 実験候補生成（制約付き）
        samples = []
        attempts = 0
        max_attempts = sample_size * 20

        while len(samples) < sample_size and attempts < max_attempts:
            row = []
            sample_dict = {}
            for col in df_bounds.columns:
                try:
                    lower = float(df_bounds.loc['lower', col])
                    upper = float(df_bounds.loc['upper', col])
                    if lower >= upper:
                        raise ValueError("下限値が上限値以上です")
                    val = np.random.uniform(lower, upper)
                    row.append(val)
                    sample_dict[col] = val
                except Exception as e:
                    st.error(f"列 '{col}' の生成範囲に問題があります: {e}")
                    row.append(np.nan)

            if is_valid_sample(sample_dict, feature_constraints, sum_constraint_expr, symbol_map, enable_sum_constraint):
                samples.append(row)

            attempts += 1

        df_samples = pd.DataFrame(samples, columns=df_bounds.columns)
        with st.expander(f"✅ 制約を満たすサンプル（{len(df_samples)}件）を表示"):
            st.dataframe(df_samples)
        
        default_name = "generated_samples.csv"
        filename = st.text_input("💬 保存するファイル名（例: my_samples.csv）", value=default_name, key="filename_tab1")
        if filename:
            st.markdown(generate_download_link(df_samples, filename), unsafe_allow_html=True)
                    
with tab2:
    st.header("② 実験候補選択（D最適基準）")
    uploaded_samples = st.file_uploader("📤 説明変数のみのCSVをアップロード", type="csv", key="samples2")
    select_size = st.number_input("🔢 選択するサンプル数", min_value=2, max_value=1000, value=10)

    if uploaded_samples:
        df_all = pd.read_csv(uploaded_samples)
        with st.expander("📊 読み込んだ候補データを表示"):
            st.dataframe(df_all)

        # 標準化
        X_all = StandardScaler().fit_transform(df_all.values)

        best_score = -np.inf
        best_subset = None

        for _ in range(100):
            subset = df_all.sample(n=select_size)
            X = StandardScaler().fit_transform(subset.values)
            score = np.linalg.det(X.T @ X)
            if score > best_score:
                best_score = score
                best_subset = subset

        # ✅ 非選択サンプルの抽出
        df_unselected = df_all.drop(best_subset.index)

        with st.expander(f"✅ 選択された候補サンプル（{len(best_subset)}件）を表示"):
           st.dataframe(best_subset)
        with st.expander(f"📂 未選択サンプル（{len(df_unselected)}件）を表示"):
           st.dataframe(df_unselected)
        
        corr_matrix = best_subset.corr()
        
        # 📈 PCA-主成分分析処理
        scaler = StandardScaler()
        X_selected = scaler.fit_transform(best_subset.values)
        X_unselected = scaler.transform(df_unselected.values)

        pca = PCA(n_components=2)
        X_selected_pca = pca.fit_transform(X_selected)
        X_unselected_pca = pca.transform(X_unselected)

        # 📐 描画領域を2列に分割
        col1, col2 = st.columns(2)

        with col1:
          with st.expander("📊 相関ヒートマップを表示"):
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f", square=True,
                        cbar_kws={"shrink": 0.8}, linewidths=0.5, ax=ax1, annot_kws={"size": 6})
            
            ax1.tick_params(axis='x', labelsize=6)
            ax1.tick_params(axis='y', labelsize=6)

            st.pyplot(fig1)
            st.write("D最適基準での候補は各特徴量間の相関係数の絶対値が小さい（＝特徴量同士が類似しない）のがよい")

        with col2:
          with st.expander("📈 PCAによる分布比較を表示"):
            fig2, ax2 = plt.subplots(figsize=(3, 2))
            ax2.scatter(X_unselected_pca[:, 0], X_unselected_pca[:, 1], label="未選択", alpha=0.5, color="gray")
            ax2.scatter(X_selected_pca[:, 0], X_selected_pca[:, 1], label="選択済み", alpha=0.8, color="blue")
            
            # 軸ラベルとタイトルの文字サイズ
            ax2.set_title("PCAによるサンプル分布", fontsize=6)
            ax2.set_xlabel("主成分1", fontsize=4)
            ax2.set_ylabel("主成分2", fontsize=4)
            
            # 軸目盛の文字サイズ
            ax2.tick_params(axis='both', labelsize=4)

            # 凡例の文字サイズ

            ax2.legend(fontsize=4)
            st.pyplot(fig2)
            
            explained = pca.explained_variance_ratio_
            st.write(f"主成分1の寄与率: {explained[0]:.2%}")
            st.write(f"主成分2の寄与率: {explained[1]:.2%}")
            
            st.write("PCAを使って高次元の探索空間を2次元に圧縮して表示、探索が偏っていないかを確認")

        # 🔧 ファイル名入力＋拡張子補完（選択群）
        filename_selected = st.text_input("💬 選択された候補サンプルの保存名", value="selected_samples_dopt", key="filename_tab2_selected")
        if filename_selected:
            if not filename_selected.lower().endswith(".csv"):
                filename_selected += ".csv"
            st.markdown(generate_download_link(best_subset, filename_selected), unsafe_allow_html=True)

        # 🔧 ファイル名入力＋拡張子補完（非選択群）
        filename_unselected = st.text_input("💬 未選択サンプルの保存名", value="unselected_samples_dopt", key="filename_tab2_unselected")
        if filename_unselected:
            if not filename_unselected.lower().endswith(".csv"):
                filename_unselected += ".csv"
            st.markdown(generate_download_link(df_unselected, filename_unselected), unsafe_allow_html=True)

# ③ 次の実験候補選択
with tab3:
    st.header("③ 次の実験候補の選択")

    # ファイルアップロード
    uploaded_train = st.file_uploader("📤 実験済みデータ（説明変数＋目的変数）CSV", type="csv")
    uploaded_candidates = st.file_uploader("📤 未実験候補データ（説明変数のみ）CSV", type="csv")

    if uploaded_train and uploaded_candidates:
        df_train = pd.read_csv(uploaded_train, index_col=0)
        df_candidates = pd.read_csv(uploaded_candidates, index_col=0)

        # 目的変数の選択
        target_columns = st.multiselect(
            "🎯 目的変数の列名を選択（*現在全て選択してください）",
            df_train.columns[-3:], 
            default=[df_train.columns[-1]]
        )

        # 目標値の入力
        target_goals = {}
        for i, col in enumerate(target_columns):
            default_goal = df_train[col].median()
            val = st.text_input(
                f"{col} の目標値（初期は中央値）",
                value=str(round(default_goal, 3)),
                key=f"goal_tab3_{col}_{i}"
            )
            try:
                target_goals[col] = float(val)
            except:
                st.warning(f"{col} の目標値が数値ではありません")

        # モデル選択
        model_types = st.multiselect(
            "🔍 回帰モデルの選択（複数可）",
            ["ols_linear", "ols_nonlinear", "svr_linear", "svr_gaussian", "gpr_one_kernel", "gpr_kernels"],
            default=["ols_linear"]
        )

        # 候補推定に使うモデル
        selected_model_for_prediction = st.selectbox(
            "📌 候補推定に使うモデル（性能比較とは別）",
            model_types
        )

        # AD手法選択
        ad_method = st.selectbox(
            "🛡️ 適用範囲（AD=）手法の選択",
            ["knn", "ocsvm", "ocsvm_gamma_optimization"]
        )

        if st.button("🚀 次の候補を選定する"):

            # ======== データ可視化 ========
            with st.expander("📊 実験済みデータの可視化（統計量＋相関＋散布図）"):

                st.subheader("📌 実験済みデータの統計量")
                numeric_columns = df_train.select_dtypes(include="number").columns.tolist()
                stats_df = df_train[numeric_columns].describe().T
                stats_df["variance"] = df_train[numeric_columns].var()
                stats_df["sum"] = df_train[numeric_columns].sum()
                stats_df["median"] = df_train[numeric_columns].median()
                stats_df = stats_df[["mean", "median", "std", "variance", "min", "max", "sum"]]
                st.dataframe(stats_df, use_container_width=True)

                st.subheader("🔗 相関ヒートマップ")
                fig_corr = px.imshow(
                    df_train.corr(),
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    title="実験済みデータの相関ヒートマップ"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                st.subheader("📈 散布図（同変数同士はヒストグラム）")
                fig_pair = create_scatterplotmatrix(
                    df_train[numeric_columns],
                    diag='histogram',
                    height=800,
                    width=800
                )
                st.plotly_chart(fig_pair, use_container_width=True)

            # ======== 説明変数・目的変数 ========
            X = df_train.drop(columns=target_columns).values
            Y = df_train[target_columns].values
            X_pred = df_candidates.values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_pred_scaled = scaler.transform(X_pred)

            # ======== AD判定 ========
            if ad_method == "knn":
                k = min(5, len(X_scaled) - 1)
                nn_model = NearestNeighbors(n_neighbors=k)
                nn_model.fit(X_scaled)
                distances, _ = nn_model.kneighbors(X_pred_scaled)
                threshold = np.percentile(distances.mean(axis=1), 95)
                inside_ad = distances.mean(axis=1) <= threshold

            elif ad_method == "ocsvm":
                ad_model = OneClassSVM(nu=0.05, gamma='auto')
                ad_model.fit(X_scaled)
                inside_ad = ad_model.predict(X_pred_scaled) == 1

            elif ad_method == "ocsvm_gamma_optimization":
                param_grid = {"gamma": 2**np.arange(-20, 10)}
                grid = GridSearchCV(OneClassSVM(nu=0.05), param_grid, cv=5)
                grid.fit(X_scaled)
                best_model = grid.best_estimator_
                inside_ad = best_model.predict(X_pred_scaled) == 1

            # ======== モデル構築関数 ========
            def build_model(model_type):
                if model_type == "ols_linear":
                    return LinearRegression()

                elif model_type == "ols_nonlinear":
                    return Pipeline([
                       ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                       ("linear", LinearRegression())
                    ])

                elif model_type == "svr_linear":
                    # ★高速化：探索範囲を大幅に縮小
                    param_grid = {
                         "C": [0.1, 1, 10],
                         "epsilon": [0.01, 0.1, 1]
                    }
                    return GridSearchCV(SVR(kernel="linear"), param_grid, cv=3)

                elif model_type == "svr_gaussian":
                    # ★高速化：gamma を "scale" + 少数に限定
                    param_grid = {
                        "C": [1, 10],
                        "epsilon": [0.01, 0.1],
                        "gamma": ["scale", 0.1, 1]
                    }
                    return GridSearchCV(SVR(kernel="rbf"), param_grid, cv=3)

                elif model_type == "gpr_one_kernel":
                    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
                    return GaussianProcessRegressor(kernel=kernel, alpha=0.01, normalize_y=True)

                elif model_type == "gpr_kernels":
                    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
                    return GaussianProcessRegressor(kernel=kernel, alpha=0.01, normalize_y=True)

            # ======== k-fold モデル性能評価 ========
            performance_table_cv = []
            kf = KFold(n_splits=5, shuffle=True, random_state=0)

            for model_type in model_types:
                row_cv = {"モデル": model_type}

                for i, col in enumerate(target_columns):
                    y = Y[:, i]
                    r2_scores = []
                    rmse_scores = []

                    for train_idx, test_idx in kf.split(X_scaled):
                        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        model = build_model(model_type)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        r2_scores.append(r2_score(y_test, y_pred))
                        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

                    row_cv[f"{col}_R²_mean"] = round(np.mean(r2_scores), 3)
                    row_cv[f"{col}_RMSE_mean"] = round(np.mean(rmse_scores), 3)

                performance_table_cv.append(row_cv)

            performance_df_cv = pd.DataFrame(performance_table_cv)

            # ======== k-fold 性能表示 ========
            with st.expander("📊 回帰モデルの性能を表示"):
                st.subheader("📋 モデル × 目的変数の性能比較（k-fold CV）")
                st.write("R² モデルがどれだけデータの分散を説明　RMSE 予測誤差の大きさ）")
                st.write("R² は 1、RMSE は 0 に近いほど良い（5-fold クロスバリデーションの平均値）")
                st.dataframe(performance_df_cv)

            # ======== k-fold 実測値 vs 推測値 ========
            with st.expander("📈 実測値 vs 推測値（k-fold CV）"):

                tabs = st.tabs(target_columns)

                for i, col in enumerate(target_columns):
                    with tabs[i]:
                        y = Y[:, i]

                        y_true_all = []
                        y_pred_all = []

                        for train_idx, test_idx in kf.split(X_scaled):
                            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]

                            model = build_model(selected_model_for_prediction)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            y_true_all.extend(y_test)
                            y_pred_all.extend(y_pred)

                        df_plot = pd.DataFrame({"実測値": y_true_all, "推測値": y_pred_all})

                        reg = LinearRegression().fit(df_plot[["実測値"]], df_plot["推測値"])
                        x_range = np.linspace(df_plot["実測値"].min(), df_plot["実測値"].max(), 100)
                        y_pred_line = reg.predict(x_range.reshape(-1, 1))

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_plot["実測値"], y=df_plot["推測値"],
                            mode="markers", name="予測結果"
                        ))
                        fig.add_trace(go.Scatter(
                            x=x_range, y=y_pred_line,
                            mode="lines", name="Trendline", line=dict(color="blue")
                        ))
                        fig.add_shape(
                            type="line",
                            x0=x_range.min(), y0=x_range.min(),
                            x1=x_range.max(), y1=x_range.max(),
                            line=dict(color="red", dash="dash")
                        )
                        fig.update_layout(
                            title=f"{col}：実測値 vs 推測値（5-fold CV）",
                            xaxis_title="実測値",
                            yaxis_title="推測値"
                        )

                        st.plotly_chart(fig, use_container_width=True)

            # ======== 候補推定（k-fold 平均予測）========
            selected_model_type = selected_model_for_prediction

            predicted_dict = {}
            uncertainty_dict = {}
            distance_total = np.zeros(len(X_pred))

            for i, col in enumerate(target_columns):
                y = Y[:, i]

                # fold ごとの予測値を保存
                fold_preds = []
                fold_stds = []

                for train_idx, test_idx in kf.split(X_scaled):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    model = build_model(selected_model_type)
                    model.fit(X_train, y_train)

                    # GPR の場合は不確かさも取得
                    if selected_model_type.startswith("gpr"):
                        y_mean, y_std = model.predict(X_pred_scaled, return_std=True)
                        fold_preds.append(y_mean)
                        fold_stds.append(y_std)
                    else:
                        y_mean = model.predict(X_pred_scaled)
                        fold_preds.append(y_mean)
                        fold_stds.append(np.zeros_like(y_mean))

                # fold 平均
                fold_preds = np.array(fold_preds)
                fold_stds = np.array(fold_stds)

                y_pred_mean = fold_preds.mean(axis=0)
                y_pred_std = fold_stds.mean(axis=0)

                predicted_dict[col] = y_pred_mean
                uncertainty_dict[col] = y_pred_std

                # 目標値との距離
                distance_total += (y_pred_mean - target_goals[col])**2

            distance_total = np.sqrt(distance_total)

            # ======== 結果 DataFrame ========
            df_result = df_candidates.copy()
            for col in target_columns:
                df_result[f"Predicted {col}"] = predicted_dict[col]
                df_result[f"Uncertainty {col}"] = uncertainty_dict[col]

            df_result["Inside AD"] = inside_ad
            df_result["Distance to Target"] = distance_total
            df_result["使用モデル"] = selected_model_type

            df_sorted = df_result.sort_values("Distance to Target")

            # ======== 表示 ========
            with st.expander("📊 候補サンプル（目標値に近い順）を表示"):
                st.dataframe(df_sorted)
                st.caption(f"🔍 AD判定には {ad_method} を使用、推定には {selected_model_type}（k-fold 平均）を使用しています。")
                st.write("AD内（inside）であれば推定結果が信頼できる可能性が高いです。")

            # ======== CSV ダウンロード ========
            filename = st.text_input("💬 保存するファイル名", value="next_candidates.csv")
            if filename and not filename.lower().endswith(".csv"):
                filename += ".csv"

            def generate_download_link(df, filename):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 CSVをダウンロード</a>'

            st.markdown(generate_download_link(df_sorted, filename), unsafe_allow_html=True)

            
with tab4:
    st.header("④ 一括型ベイズ最適化による実験候補選択（候補あり）")

    # ファイルアップロード
    uploaded_train = st.file_uploader("📤 実験済みデータ（説明変数＋目的変数）CSV", type="csv", key="bayes_train")
    uploaded_candidates = st.file_uploader("📤 候補データ（説明変数のみ）CSV", type="csv", key="bayes_candidates")

    # モデル選択
    model_type = st.selectbox("🔧 GPRモデルの選択", ["gpr_one_kernel", "gpr_kernels"])

    # 目的変数の選択
    if uploaded_train:
        df_train = pd.read_csv(uploaded_train, index_col=0)
        target_columns = st.multiselect(
            "🎯 目的変数の列名を選択（*現在全て選択してください）",
            df_train.columns[-3:],
            default=[df_train.columns[-1]],
            key="target_columns_tab4"
        )

        # 目標値の入力
        goal_values = {}
        for i, col in enumerate(target_columns):
            default_goal = df_train[col].median()  # 中央値を初期値に設定
            val = st.text_input(f"{col} の目標値（初期は中央値）", value=str(round(default_goal, 3)), key=f"goal_tab4_{col}_{i}")
            try:
                goal_values[col] = float(val)
            except:
                st.warning(f"{col} の目標値が数値ではありません")

    # 獲得関数の選択
    acquisition = st.selectbox("📈 獲得関数の選択", ['PTR', 'PI', 'EI', 'MI'])

    if uploaded_train and uploaded_candidates and target_columns:
        df_candidates = pd.read_csv(uploaded_candidates, index_col=0)
        X_train = df_train.drop(columns=target_columns).values
        X_pred = df_candidates.values

        df_result = df_candidates.copy()

        for col in target_columns:
            y_train = df_train[col].values
            goal = goal_values[col]

            # モデル構築
            if model_type == "gpr_one_kernel":
                kernel = C(1.0) * RBF(length_scale=1.0)
            elif model_type == "gpr_kernels":
                kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

            gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.01, normalize_y=True)
            gpr.fit(X_train, y_train)
            y_pred, y_std = gpr.predict(X_pred, return_std=True)

            # 獲得関数の計算
            from scipy.stats import norm
            z = (y_pred - goal) / y_std
            if acquisition == "PTR":
                score = norm.cdf(z)
            elif acquisition == "PI":
                score = norm.cdf(z)
            elif acquisition == "EI":
                score = (y_pred - goal) * norm.cdf(z) + y_std * norm.pdf(z)
            elif acquisition == "MI":
                score = -np.log(y_std + 1e-9)  # 情報量最大化（不確かさの逆）

            # 結果格納
            df_result[f"Predicted {col}"] = y_pred
            df_result[f"Uncertainty {col}"] = y_std
            df_result[f"{acquisition} {col}"] = score
            df_result[f"Distance to Goal {col}"] = np.abs(y_pred - goal)  

        # スコア合算（目的変数が複数ある場合）
        score_cols = [f"{acquisition} {col}" for col in target_columns]
        df_result["Score Sum"] = df_result[score_cols].sum(axis=1)

        df_sorted = df_result.sort_values("Score Sum", ascending=False)

        # 📊 ベイズ最適化による候補スコア一覧（折り畳み表示）
        with st.expander("📊 ベイズ最適化による候補スコア一覧（高スコア順）を表示"):
           st.dataframe(df_sorted)

        # 散布図表示（目的変数ごと）
        # 説明変数の列名を定義（目的変数や中間指標を除外）
        with st.expander("📈 推測値 vs 目標距離 vs EI の散布図を表示"):
           # 🔧 目的変数に応じて除外列を定義

           exclude_cols = ["Score Sum"]
           for col in target_columns:
              exclude_cols += [
                f"Predicted {col}",
                f"Uncertainty {col}",
                f"{acquisition} {col}",
                f"Distance to Goal {col}"
           ]

           # hover_data に使う説明変数を抽出
           hover_vars = [col for col in df_result.columns if col not in exclude_cols]
           
           for col in target_columns:
              fig = px.scatter(
                  df_result,
                  x=f"Predicted {col}",
                  y=f"Distance to Goal {col}",
                  size=f"Uncertainty {col}",
                  color=f"{acquisition} {col}",
                  hover_data=hover_vars,
                  title=f"{col}：推測値 vs 目標距離",
                  labels={
                     f"Predicted {col}": "推測値",
                     f"Distance to Goal {col}": "目標との距離",
                     f"Uncertainty {col}": "予測の不確かさ",
                     f"{acquisition} {col}": f"獲得関数（{acquisition}）"

                   }
              )
              st.plotly_chart(fig, use_container_width=True, key=f"scatter_{col}")        
               
        # CSVダウンロード
        filename = st.text_input("💬 保存するファイル名", value="bayesian_candidates.csv", key="filename_tab4")
        if filename and not filename.lower().endswith(".csv"):
            filename += ".csv"

        def generate_download_link(df, filename):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 CSVをダウンロード</a>'

        st.markdown(generate_download_link(df_sorted, filename), unsafe_allow_html=True)

with tab5:
    st.header("⑤ 逐次型ベイズ最適化による実験候補選択（候補なし）")

    # 実験済みデータのアップロード
    uploaded_train = st.file_uploader("📤 実験済みデータ（説明変数＋目的変数）CSV", type="csv", key="seq_train")

    if uploaded_train:
        df_train = pd.read_csv(uploaded_train, index_col=0)

        # 列の分類
        all_columns = df_train.columns.tolist()
        target_columns = st.multiselect("🎯 目的変数の列名（複数選択可）", all_columns, default=all_columns[-1:], key="target_multi")
        feature_columns = st.multiselect("🔍 説明変数の列名", [col for col in all_columns if col not in target_columns], default=all_columns[:-1], key="features_multi")

        # 探索空間の定義
        search_space = {}

        with st.expander("📐 探索空間の定義（クリックで展開）", expanded=False):
            for col in feature_columns:
                c1, c2 = st.columns(2)
                with c1:
                    min_val = st.number_input(f"{col} の最小値", value=float(df_train[col].min()), key=f"min_{col}")
                with c2:
                    max_val = st.number_input(f"{col} の最大値", value=float(df_train[col].max()), key=f"max_{col}")
                search_space[col] = (min_val, max_val)

        # 目標値の指定
        with st.expander("🎯 目標値　獲得関数　探索　実験提案数の設定"):
            goal_values = {}
            for col in target_columns:
                goal_values[col] = st.number_input(f"{col} の目標値", value=float(df_train[col].median()), key=f"goal_{col}")
        
        # 可視化対象の変数選択（1〜2個）
        # st.subheader("📊 可視化対象の説明変数")
        # plot_vars = st.multiselect("可視化に使う変数（1〜2個）", feature_columns, max_selections=2)

        # # 固定する変数の値をスライダーで指定
        # fixed_values = {}
        # for col in feature_columns:
        #     if col not in plot_vars:
        #         fixed_values[col] = st.slider(f"{col} の固定値", min_value=search_space[col][0], max_value=search_space[col][1],
        #                                     value=(search_space[col][0] + search_space[col][1]) / 2)  
        
        
            acq_method = st.selectbox("📈 獲得関数の選択", ["PTR", "PI", "EI", "MI"], key="acquisition_method_selector")
            n_calls = st.number_input("🔁 探索ステップ数（推測値で仮実験）", min_value=1, max_value=20, value=5, step=1)
            n_suggestions = st.number_input("🔢 提案する実験条件の数", min_value=1, max_value=20, value=5, step=1)
                
               
        # 最適点探索
        if st.button("🚀 次の実験候補を提案", key="run_multi_bayes"):
            
            # GPRモデル構築（目的変数ごと）
            X_train = df_train[feature_columns].values
            gpr_models = {}
            for col in target_columns:
                y_train = df_train[col].values
                kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.01, normalize_y=True)
                gpr.fit(X_train, y_train)
                gpr_models[col] = gpr

            # 合成獲得関数（加重平均）
            def acquisition_function(x):
                x = np.array(x).reshape(1, -1)
                total_score = 0
                delta = 0.1  # 許容幅（必要に応じて調整）

                for col in target_columns:
                    gpr = gpr_models[col]
                    goal = goal_values[col]
                    y_pred, y_std = gpr.predict(x, return_std=True)
                    
                    if acq_method == "EI":
                        z = (y_pred - goal) / (y_std + 1e-9)
                        score = (y_pred - goal) * norm.cdf(z) + y_std * norm.pdf(z)
                    elif acq_method == "PI":
                        z = (y_pred - goal) / (y_std + 1e-9)
                        score = norm.cdf(z)
                    elif acq_method == "MI":
                        score = np.log(y_std + 1e-9)
                    elif acq_method == "PTR":
                        lower = goal - delta
                        upper = goal + delta
                        score = norm.cdf((upper - y_pred) / (y_std + 1e-9)) - norm.cdf((lower - y_pred) / (y_std + 1e-9))
                    else:
                        score = 0  # fallback

                    total_score += score[0]
                    return -total_score

            # 初期点・制約
            x0 = [(search_space[col][0] + search_space[col][1]) / 2 for col in feature_columns]
            bounds = [search_space[col] for col in feature_columns]

            # 逐次探索ループ
            suggestions = []
            for step in range(n_suggestions):
                # GPRモデル再構築（毎ステップ）
                X_train = df_train[feature_columns].values
                gpr_models = {}
                for col in target_columns:
                    y_train = df_train[col].values
                    kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
                    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.01, normalize_y=True)
                    gpr.fit(X_train, y_train)
                    gpr_models[col] = gpr

                # 汎用獲得関数（EI / PI / PI / PTR 対応）
                def acquisition_multi(x):
                    x = np.array(x).reshape(1, -1)
                    total_score = 0
                    delta = 0.1  # PTR用の許容幅（必要に応じて調整）

                    for col in target_columns:
                        gpr = gpr_models[col]
                        goal = goal_values[col]
                        y_pred, y_std = gpr.predict(x, return_std=True)

                        if acq_method == "EI":
                            z = (y_pred - goal) / (y_std + 1e-9)
                            score = (y_pred - goal) * norm.cdf(z) + y_std * norm.pdf(z)

                        elif acq_method == "PI":
                            z = (y_pred - goal) / (y_std + 1e-9)
                            score = norm.cdf(z)

                        elif acq_method == "MI":
                            score = np.log(y_std + 1e-9)

                        elif acq_method == "PTR":
                            lower = goal - delta
                            upper = goal + delta
                            score = norm.cdf((upper - y_pred) / (y_std + 1e-9)) - norm.cdf((lower - y_pred) / (y_std + 1e-9))

                        else:
                            score = 0  # fallback

                        total_score += score[0]

                    return -total_score
                
                # 最適化
                x0 = [(search_space[col][0] + search_space[col][1]) / 2 for col in feature_columns]
                bounds = [search_space[col] for col in feature_columns]
                result = minimize(acquisition_function, x0=x0, bounds=bounds, method="L-BFGS-B")
                next_x = result.x

                # 推測値の取得と仮実験データ追加
                new_row = {}
                for i, col in enumerate(feature_columns):
                    new_row[col] = next_x[i]
                for col in target_columns:
                    gpr = gpr_models[col]
                    y_pred = gpr.predict(next_x.reshape(1, -1))[0]
                    new_row[col] = y_pred

                df_train = pd.concat([df_train, pd.DataFrame([new_row])], ignore_index=True)
                suggestions.append(new_row)      

            # 最終ステップの提案点表示
            with st.expander("📋 提案された実験条件一覧（推測値ベース）"):
                st.dataframe(pd.DataFrame(suggestions))
            
            suggestions_df = pd.DataFrame(suggestions)
          
            # ファイル名入力欄（.csvが付いていなければ追加）
            filename = st.text_input("💬 保存するファイル名", value="suggested_experiments.csv", key="filename_suggestions")
            if filename and not filename.lower().endswith(".csv"):
                filename += ".csv"

            def generate_download_link(df, filename):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 CSVをダウンロード</a>'
                return href

            st.markdown(generate_download_link(suggestions_df, filename), unsafe_allow_html=True)

            # # 🔍 各目的変数ごとの描画処理
            # with st.expander(f"📊 {acq_method} 可視化（目的変数ごと）", expanded=False):
            #     for col in target_columns:
            #         st.subheader(f"📈 {col} の {acq_method} 可視化")

            #         gpr = gpr_models[col]
            #         goal = goal_values[col]

            #         if len(plot_vars) == 1:
            #             var = plot_vars[0]
            #             x_range = np.linspace(search_space[var][0], search_space[var][1], 100)
            #             X_plot = []
            #             for x in x_range:
            #                 row = []
            #                 for col_f in feature_columns:
            #                     if col_f == var:
            #                         row.append(x)
            #                     else:
            #                         row.append(fixed_values[col_f])
            #                 X_plot.append(row)
            #             X_plot = np.array(X_plot)

            #             y_mean, y_std = gpr.predict(X_plot, return_std=True)
            #             z = (y_mean - goal) / (y_std + 1e-9)
            #             if acq_method == "EI":
            #                 z = (y_mean - goal) / (y_std + 1e-9)
            #                 acq_curve = (y_mean - goal) * norm.cdf(z) + y_std * norm.pdf(z)
            #             elif acq_method == "PI":
            #                 z = (y_mean - goal) / (y_std + 1e-9)
            #                 acq_curve = norm.cdf(z)
            #             elif acq_method == "UCB":
            #                 kappa = 2.0
            #                 acq_curve = y_mean + kappa * y_std

            #             fig, ax1 = plt.subplots()
            #             ax1.plot(x_range, y_mean, 'k-', label='GP Mean')
            #             ax1.fill_between(x_range, y_mean - y_std, y_mean + y_std, alpha=0.2, color='blue', label='Uncertainty')
            #             ax1.set_ylabel(f"{col} 推測値")
            #             ax2 = ax1.twinx()
            #             ax2.plot(x_range, acq_curve, 'r-', label=f'{acq_method}')
            #             ax2.set_ylabel(acq_method)
            #             fig.legend(loc="upper left")
            #             st.pyplot(fig)

            #         elif len(plot_vars) == 2:
            #             var1, var2 = plot_vars
            #             x_range = np.linspace(search_space[var1][0], search_space[var1][1], 50)
            #             y_range = np.linspace(search_space[var2][0], search_space[var2][1], 50)
            #             X1, X2 = np.meshgrid(x_range, y_range)
            #             X_plot = []
            #             for i in range(len(X1.flatten())):
            #                 row = []
            #                 for col_f in feature_columns:
            #                     if col_f == var1:
            #                         row.append(X1.flatten()[i])
            #                     elif col_f == var2:
            #                         row.append(X2.flatten()[i])
            #                     else:
            #                         row.append(fixed_values[col_f])
            #                 X_plot.append(row)
            #             X_plot = np.array(X_plot)

            #             y_mean, y_std = gpr.predict(X_plot, return_std=True)
            #             z = (y_mean - goal) / (y_std + 1e-9)
            #             if acq_method == "EI":
            #                 z = (y_mean - goal) / (y_std + 1e-9)
            #                 acq_curve = (y_mean - goal) * norm.cdf(z) + y_std * norm.pdf(z)
            #             elif acq_method == "PI":
            #                 z = (y_mean - goal) / (y_std + 1e-9)
            #                 acq_curve = norm.cdf(z)
            #             elif acq_method == "UCB":
            #                 kappa = 2.0
            #                 acq_curve = y_mean + kappa * y_std

            #             fig, ax = plt.subplots()
            #             acq_grid = acq_curve.reshape(X1.shape)
            #             sns.heatmap(acq_grid, xticklabels=False, yticklabels=False, cmap="Reds", cbar_kws={'label': 'EI'}, ax=ax)
            #             ax.set_title(f"{var1} vs {var2} の {acq_method} ヒートマップ（{col}）")
            #             st.pyplot(fig)       
with tab6:

    import numpy as np
    import pandas as pd
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from scipy.stats import multivariate_normal
    import optuna
    import streamlit as st

    # ================================
    # 共分散行列の取得
    # ================================
    def extract_covariance(gmm, k):
        cov_type = gmm.covariance_type
        if cov_type == 'full':
            return gmm.covariances_[k]
        elif cov_type == 'diag':
            return np.diag(gmm.covariances_[k])
        elif cov_type == 'spherical':
            return np.eye(gmm.means_.shape[1]) * gmm.covariances_[k]
        elif cov_type == 'tied':
            return gmm.covariances_
        else:
            raise ValueError(f"Unknown covariance_type: {cov_type}")

    # ================================
    # GMR（標準式 + NaN防止）
    # ================================
    def gmr_predict(gmm, x_query, input_idx, output_idx):
        x_query = np.array(x_query).reshape(-1)

        K = gmm.n_components
        means = gmm.means_
        pis = gmm.weights_

        y_means = []
        y_vars = []
        weights = []

        for k in range(K):
            mu = means[k]
            cov = extract_covariance(gmm, k)

            mu_x = mu[input_idx]
            mu_y = mu[output_idx]

            Sigma_xx = cov[np.ix_(input_idx, input_idx)]
            Sigma_xy = cov[np.ix_(input_idx, output_idx)]
            Sigma_yx = cov[np.ix_(output_idx, input_idx)]
            Sigma_yy = cov[np.ix_(output_idx, output_idx)]

            # --- 数値安定化 ---
            Sigma_xx = Sigma_xx + np.eye(Sigma_xx.shape[0]) * 1e-6
            inv_Sigma_xx = np.linalg.pinv(Sigma_xx)

            # 条件付き平均
            y_k = mu_y + Sigma_yx @ inv_Sigma_xx @ (x_query - mu_x)

            # 条件付き分散
            cov_k = Sigma_yy - Sigma_yx @ inv_Sigma_xx @ Sigma_xy

            # posterior weight（NaN防止）
            px = pis[k] * multivariate_normal.pdf(
                x_query, mean=mu_x, cov=Sigma_xx, allow_singular=True
            )
            weights.append(px)

            y_means.append(y_k)
            y_vars.append(cov_k)

        weights = np.array(weights)
        weights /= weights.sum()

        y_mean = np.sum([w * m for w, m in zip(weights, y_means)], axis=0)
        y_var = np.sum([w * v for w, v in zip(weights, y_vars)], axis=0)

        return y_mean, y_var

    # ================================
    # GMM 学習
    # ================================
    def fit_gmm(XY, n_components=3, covariance_type='full'):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=42
        )
        gmm.fit(XY)
        return gmm

    # ================================
    # Optuna 目的関数（NaN完全防止版）
    # ================================
    def objective(trial, XY, input_idx, output_idx):
        n_components = trial.suggest_int("n_components", 2, 6)
        covariance_type = trial.suggest_categorical(
            "covariance_type", ["full", "diag", "tied", "spherical"]
        )

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in kf.split(XY):
            gmm = fit_gmm(XY[train_idx], n_components, covariance_type)

            preds = []
            for x in XY[test_idx][:, input_idx]:
                pred, _ = gmr_predict(gmm, x, input_idx, output_idx)
                preds.append(pred)

            preds = np.array(preds)

            # NaN があれば trial を無効にする
            if np.isnan(preds).any():
                return -9999

            y_true = XY[test_idx][:, output_idx]

            # 多次元 R²（安定版）
            score = r2_score(
                y_true,
                preds,
                multioutput="uniform_average"
            )
            scores.append(score)

        return np.mean(scores)

    # ================================
    # Streamlit UI
    # ================================
    st.header("🔍 GMRによる逆解析（多目的対応・複数候補生成）")

    uploaded_file = st.file_uploader("📁 実験結果のCSVファイルをアップロード", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.select_dtypes(include=[np.number])

        all_columns = df.columns.tolist()

        st.subheader("📌 変数選択")
        target_vars = st.multiselect("目的変数を選択（1〜5個）", all_columns)
        input_vars = [col for col in all_columns if col not in target_vars]

        input_idx = [df.columns.get_loc(v) for v in target_vars]
        output_idx = [df.columns.get_loc(v) for v in input_vars]

        st.subheader("🎯 目標値入力")
        target_values = {
            var: st.number_input(f"{var} の目標値", value=float(df[var].mean()))
            for var in target_vars
        }

        num_candidates = st.number_input(
            "生成する候補数（例：100〜300）", min_value=20, max_value=500, value=150
        )

        top_k = st.number_input(
            "表示する上位候補数", min_value=3, max_value=50, value=10
        )

        if st.button("🚀 推定開始"):
            XY = df.values
            x_query = np.array([target_values[v] for v in target_vars])

            with st.spinner("Optuna でハイパーパラメータ最適化中..."):
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    lambda trial: objective(trial, XY, input_idx, output_idx),
                    n_trials=15
                )

            best_params = study.best_params
            gmm = fit_gmm(XY, **best_params)

            y_center, y_var = gmr_predict(gmm, x_query, input_idx, output_idx)

            # ================================
            # 複数候補生成（中心点 ± ランダム揺らぎ）
            # ================================
            candidates = []
            for _ in range(num_candidates):
                noise = np.random.multivariate_normal(
                    mean=np.zeros(len(output_idx)),
                    cov=y_var * 0.5 + np.eye(len(output_idx)) * 1e-6
                )
                cand = y_center + noise
                candidates.append(cand)

            candidates = np.array(candidates)

            # ================================
            # 中心点からの距離でソート
            # ================================
            distances = np.linalg.norm(candidates - y_center, axis=1)
            sorted_idx = np.argsort(distances)

            top_candidates = candidates[sorted_idx][:top_k]

            result_df = pd.DataFrame(top_candidates, columns=input_vars)

            st.success("✅ 推定完了！")

            st.subheader("📊 上位候補（目標値に近い順）")
            st.dataframe(result_df)

            st.download_button(
                "📥 CSVをダウンロード",
                result_df.to_csv(index=False),
                file_name="gmr_candidates.csv"
            )

            st.subheader("📈 ハイパーパラメータ選定根拠")
            st.markdown(f"""
            - **最適な混合数**: {best_params['n_components']}
            - **分散共分散の種類**: {best_params['covariance_type']}
            - **決定係数 R²**: {study.best_value:.4f}
            """)
