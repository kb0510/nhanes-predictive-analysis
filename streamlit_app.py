"""NIT6160 depression project talk: the 18 presentation slides, with interactive 3D charts.

Run:  streamlit run streamlit_app.py
Data: data/nhanes_clean.csv (built by depression_analysis.ipynb, Part A).
Notes: notes.json = the speaker notes of the Slides deck, slide by slide.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title='Predicting Depression: NIT6160 Talk', layout='wide')

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'nhanes_clean.csv'
NOTES = json.load(open(HERE / 'notes.json', encoding='utf-8'))
NUMERIC = ['age', 'income_ratio', 'sleep_hours', 'bmi', 'sitting_minutes']
CATEGORIES = ['sex', 'education', 'marital_status', 'sleep_trouble', 'smoked_100', 'drank_past_year',
              'vigorous_activity', 'heart_disease', 'cancer_ever', 'told_overweight']
NAVY, ORANGE, LIGHT_ORANGE, GREY, RED, BLUE, GREEN = '#14213D', '#C2611F', '#F4A259', '#9aa5b1', '#d62728', '#1f77b4', '#2ca02c'
GROUP_COLOURS = {'Young, lower income': BLUE, 'Older, lower income': '#ff7f0e', 'Higher income': GREEN}

# deck look: navy headings, orange accents, light background cards
st.markdown(f"""<style>
h1, h2, h3 {{ color: {NAVY}; }}
.eyebrow {{ color: {ORANGE}; letter-spacing: 3px; text-transform: uppercase; font-weight: 600; font-size: 0.9rem; }}
.big {{ font-size: 3.2rem; font-weight: 700; color: {NAVY}; line-height: 1.1; }}
.card {{ background: #ffffff; border: 1px solid #E1DED6; border-radius: 14px; padding: 18px 22px; height: 100%; }}
.dark {{ background: {NAVY}; color: #F7F6F2; border-radius: 14px; padding: 22px 28px; }}
.dark b {{ color: {LIGHT_ORANGE}; }}
</style>""", unsafe_allow_html=True)


# ---------- data and models (cached: computed once, reused on every slide) ----------
@st.cache_data
def load():
    df = pd.read_csv(DATA)
    df['status'] = np.where(df['depressed'] == 1, 'depressed', 'not depressed')
    df['income_group'] = pd.cut(df['income_ratio'], bins=[-1, 1, 2, 4, 5],
                                labels=['Below poverty line', '1–2×', '2–4×', '4× or more'])
    df['age_group'] = pd.cut(df['age'], bins=[17, 29, 44, 59, 80], labels=['18–29', '30–44', '45–59', '60–80'])
    return df


@st.cache_resource
def models(df):
    """Steps 13-17: 80/20 stratified split, prep pipeline, the 3 tuned classifiers + the plain (unbalanced) one."""
    X, y = df[NUMERIC + CATEGORIES], df['depressed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    prep = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORIES)])
    tuned = {
        'Logistic regression': LogisticRegression(C=0.01, class_weight='balanced', max_iter=2000),
        'Random forest': RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=1,
                                                class_weight='balanced', random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', C=1, gamma=0.01, class_weight='balanced'),
        'LR, not balanced': LogisticRegression(max_iter=2000)}
    fitted = {name: Pipeline([('prep', prep), ('model', m)]).fit(X_train, y_train) for name, m in tuned.items()}
    return fitted, X_test, y_test.to_numpy()


@st.cache_data
def clusters(df):
    """Step 21: median fill, min-max scale, elbow k = 1-8, k-means k = 3 (same settings as the notebook)."""
    C = df[NUMERIC].fillna(df[NUMERIC].median())
    scaled = (C - C.min()) / (C.max() - C.min())
    wss = {k: KMeans(n_clusters=k, n_init=20, random_state=0).fit(scaled).inertia_ for k in range(1, 9)}
    km = KMeans(n_clusters=3, n_init=20, random_state=0).fit(scaled)
    centres = pd.DataFrame(km.cluster_centers_, columns=NUMERIC)
    rich = centres['income_ratio'].idxmax()
    older = centres.drop(rich)['age'].idxmax()
    names = {c: 'Higher income' if c == rich else 'Older, lower income' if c == older else 'Young, lower income'
             for c in range(3)}
    real = centres * (C.max() - C.min()) + C.min()
    real.index = [names[c] for c in range(3)]
    # Step 21j: 3 PCA axes of the same scaled columns, and the group centres moved into that space
    pca = PCA(n_components=3, random_state=0)
    xyz = pd.DataFrame(pca.fit_transform(scaled), columns=['PC1', 'PC2', 'PC3'], index=df.index)
    cent = pd.DataFrame(pca.transform(centres), columns=['PC1', 'PC2', 'PC3'], index=real.index)
    return pd.Series(km.labels_, index=df.index).map(names), real, wss, xyz, cent, pca.explained_variance_ratio_


df = load()
overall = df['depressed'].mean() * 100
fitted, X_test, y_test = models(df)
groups, centres, wss, xyz, pca_centres, pca_share = clusters(df)


def score(name):
    """Scores for the ROC curve: probability where the model has one, else the SVM decision value."""
    m = fitted[name]
    return m.predict_proba(X_test)[:, 1] if hasattr(m, 'predict_proba') else m.decision_function(X_test)


def counts(name, cut=None):
    guess = (score(name) >= cut) if cut is not None else fitted[name].predict(X_test)
    tp, fp = int((guess & (y_test == 1)).sum()), int((guess & (y_test == 0)).sum())
    return dict(TP=tp, FN=92 - tp, FP=fp, TN=922 - fp)


# ---------- chart helpers ----------
def cube(x, y, h, colour, w=0.6, d=0.6, opacity=0.92):
    """One 3D bar: a box of height h standing on (x, y)."""
    xs = [x - w / 2, x + w / 2, x + w / 2, x - w / 2] * 2
    ys = [y - d / 2, y - d / 2, y + d / 2, y + d / 2] * 2
    zs = [0] * 4 + [h] * 4
    return go.Mesh3d(x=xs, y=ys, z=zs, color=colour, opacity=opacity, flatshading=True, showlegend=False,
                     i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                     k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], hoverinfo='skip')


def label3d(x, y, z, text, size=12, colour=NAVY):
    return go.Scatter3d(x=[x], y=[y], z=[z], mode='text', text=[text], showlegend=False, hoverinfo='skip',
                        textfont=dict(size=size, color=colour))


def plane(x0, x1, y0, y1, z, colour='gray', opacity=0.22):
    return go.Surface(x=[x0, x1], y=[y0, y1], z=[[z, z], [z, z]], opacity=opacity, showscale=False,
                      colorscale=[[0, colour], [1, colour]], hoverinfo='skip')


def show(fig, h=560):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=30, b=0), legend=dict(itemsizing='constant'),
                      paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, width='stretch')


def stat(col, number, text, colour=ORANGE):
    col.markdown(f"<div class='card'><div class='big' style='color:{colour}'>{number}</div>{text}</div>",
                 unsafe_allow_html=True)


def tip():
    st.caption('3D chart: drag to rotate · scroll to zoom · hover for values · double-click to reset')


# ---------- the 19 slides (same order and content as the Slides deck) ----------
def cover():
    st.markdown("<div class='eyebrow'>NIT6160 Data Warehousing and Mining · Project</div>", unsafe_allow_html=True)
    st.title('Predicting depression from everyday information')
    st.subheader('A machine learning analysis of NHANES 2017–2018')
    st.write('Kushal Bansal · s8269160 · Term H8W4, 2026')


def problem():
    st.markdown("<div class='eyebrow'>The problem</div>", unsafe_allow_html=True)
    st.title('Depression is common, but many people who have it are never identified.')
    st.markdown('### Screening with the PHQ-9 only happens if someone asks.\n'
                '### **Can sleep, income and lifestyle point to who should be screened?**')


def dataset():
    st.header('The dataset: NHANES 2017–2018')
    left, right = st.columns([1.4, 1])
    left.table(pd.DataFrame({'Group': ['Demographics', 'Lifestyle', 'Health', 'Class'],
                             'Variables': ['age, sex, education, marital status, income ratio',
                                           'sleep hours, trouble sleeping, smoking, alcohol, vigorous activity, sitting',
                                           'BMI, heart disease, cancer, told overweight',
                                           'depressed = PHQ-9 score of 10 or more']}).set_index('Group'))
    left.caption('US government health survey, free and public. 8 files joined on the person ID.')
    a, b = right.columns(2)
    stat(a, f'{len(df):,}', 'adults', NAVY)
    stat(b, len(NUMERIC + CATEGORIES), 'predictors', NAVY)
    right.write('')
    right.markdown(f"<div class='dark'><div class='big' style='color:{LIGHT_ORANGE}'>{overall:.1f}%</div>"
                   f"are depressed ({int(df['depressed'].sum())} adults)</div>", unsafe_allow_html=True)


def rqs():
    st.header('Five research questions')
    for n, q in [(1, 'Which **demographic** factors are associated with depression?'),
                 (2, 'How are **lifestyle** factors (sleep, smoking, alcohol, activity, sitting) related to it?'),
                 (3, 'Can **machine learning** predict who screens positive, and which model works best?'),
                 (4, 'Which **variables** contribute most to the predictions?'),
                 (5, 'Do natural **groups** of adults, found without the outcome, show different rates?')]:
        st.markdown(f"<div class='card' style='margin-bottom:10px'><b style='color:{ORANGE};font-size:1.3rem'>RQ{n}</b>"
                    f"&nbsp;&nbsp;{q.replace('**', '')}</div>", unsafe_allow_html=True)


# histogram settings per column: readable name, bin width, one-line reading
HIST = {
    'phq9_score': ('PHQ-9 score (0–27)', 1, 'About a third of adults score 0; only the tail from 10 up (orange) counts as depressed.'),
    'age': ('Age (years)', 5, 'Spread evenly from 18 to 80; the tall last bar is everyone aged 80 or over.'),
    'income_ratio': ('Income ratio (× poverty line)', 0.25, 'About half are between 0.5 and 3× the poverty line; the tall bar at 5 is everyone at 5× or more.'),
    'sleep_hours': ('Weekday sleep (hours)', 1, 'Bell-shaped around 7–8 hours, with a few very short and very long sleepers.'),
    'bmi': ('BMI (kg/m²)', 2, 'Peaks at 25–30, with a long right tail of high values.'),
    'sitting_minutes': ('Sitting (minutes a day)', 60, 'Spikes at round hours: people answer in whole hours.'),
}


def explore():
    st.header('Exploring the data first')
    left, right = st.columns([2.2, 1])
    col = left.selectbox('Histogram of', list(HIST), format_func=lambda c: HIST[c][0])
    label, width, reading = HIST[col]
    values = df[col].dropna()
    # bins start on a multiple of the width, so bars line up with whole scores, ages and hours
    lo = np.floor(values.min() / width) * width
    starts = np.floor((values - lo) / width) * width + lo
    counts = starts.value_counts().sort_index()
    discrete = col == 'phq9_score'                      # one bar per whole score, centred on it
    xs = counts.index if discrete else counts.index + width / 2
    colours = [ORANGE if discrete and s >= 10 else NAVY for s in counts.index]
    fig = go.Figure(go.Bar(x=xs, y=counts.values, width=width * 0.92, marker_color=colours, showlegend=False,
                           hovertemplate=f'{label}: %{{x}}<br>adults: %{{y:,}}<extra></extra>'))
    top = counts.max()
    # mean and median, labelled at different heights so they never overlap
    for value, name, dash, y in [(values.median(), 'median', 'solid', 1.0), (values.mean(), 'mean', 'dot', 0.9)]:
        fig.add_vline(value, line_color=GREY if name == 'mean' else '#E8A33D', line_dash=dash, line_width=2)
        fig.add_annotation(x=value, y=y, yref='paper', text=f'{name} {value:.1f}', showarrow=False,
                           xanchor='left', xshift=4, font=dict(size=12, color='#5A6578'))

    def callout(value, text, ax, ay=-30):
        # point at the bar that holds this value (its bin start, then the bar centre)
        start = np.floor((value - lo) / width) * width + lo
        i = int(np.argmin(np.abs(np.asarray(counts.index) - start)))
        fig.add_annotation(x=xs[i], y=counts.iloc[i], text=text, showarrow=True, arrowhead=2, arrowcolor=ORANGE,
                           ax=ax, ay=ay, bgcolor='white', bordercolor=ORANGE, font=dict(size=12, color=NAVY))

    if discrete:
        fig.add_vrect(x0=9.5, x1=values.max() + 0.5, fillcolor=ORANGE, opacity=0.07, line_width=0)
        fig.add_annotation(x=18, y=top * 0.45, text=f'depressed (≥ 10): {overall:.1f}% of adults', showarrow=False,
                           font=dict(size=13, color=ORANGE))
        callout(0, f"{(values == 0).mean() * 100:.0f}% score 0", 110, 60)
        fig.update_xaxes(range=[-0.7, values.max() + 0.7])
    elif col == 'age':
        callout(80, f"{(values == 80).sum()} adults coded 80 (80 or older)", -130)
    elif col == 'income_ratio':
        callout(5, f"{(values == 5).sum()} adults capped at 5×", -110)
    elif col == 'bmi':
        fig.add_annotation(x=60, y=top * 0.25, text='long right tail', showarrow=False, font=dict(size=12, color=ORANGE))
    fig.update_layout(xaxis_title=label, yaxis_title='adults', bargap=0, plot_bgcolor='white',
                      yaxis=dict(gridcolor='#E3E8EF'), xaxis=dict(showgrid=False))
    with left:
        show(fig, 420)
        st.caption(reading)
    missing = df['income_ratio'].isna().mean() * 100
    dropped = int(df.loc[df[NUMERIC + CATEGORIES].isna().any(axis=1), 'depressed'].sum())
    stat(right, f'{missing:.1f}%', 'income missing, the most of any column')
    right.write('')
    stat(right, f'{dropped} of {int(df["depressed"].sum())}', 'depressed adults kept by filling gaps, not dropping rows')
    right.write('')
    stat(right, 'Kept', 'outliers (BMI 141, 11+ h sleep) are real people')


def boxplots():
    """Box plots + correlation on one slide (the deck merged them: both show weak, overlapping signals)."""
    st.header('Weak, overlapping signals')
    left, right = st.columns([1.2, 1])
    with left:
        col = st.radio('Box plot of', NUMERIC, index=1, horizontal=True)
        fig = px.box(df, x='status', y=col, color='status', points='outliers',
                     color_discrete_map={'depressed': RED, 'not depressed': GREY},
                     category_orders={'status': ['not depressed', 'depressed']})
        fig.update_layout(showlegend=False, xaxis_title='')
        show(fig, 430)
        med = df.groupby('status')[col].median()
        st.caption(f'{col}: median {med["not depressed"]:g} (not depressed) vs {med["depressed"]:g} (depressed)')
    fig, corr = corr_figure()
    with right:
        st.markdown('**Correlations in 3D** (bar height = strength, orange = income vs PHQ-9)')
        show(fig, 470)
        tip()
    income_med = df.groupby('status')['income_ratio'].median()
    long_sleep = df.assign(l=df['sleep_hours'] >= 10).groupby('status')['l'].mean() * 100
    off = corr.loc[NUMERIC, NUMERIC].where(~np.eye(5, dtype=bool)).abs().max().max()
    a, b, c, d = st.columns(4)
    stat(a, f"{income_med['depressed']:.2f} vs {income_med['not depressed']:.2f}", 'median income ratio, depressed vs not')
    stat(b, f"r = {corr.loc['income_ratio', 'phq9_score']:.2f}", 'income vs PHQ-9: the strongest link, still weak')
    stat(c, f"{long_sleep['depressed']:.0f}% vs {long_sleep['not depressed']:.0f}%", 'sleep 10+ hours; but sleep r ≈ 0 (U-shape)')
    stat(d, 'Overlap', f'in every box; max r between features {off:.2f}', NAVY)


def patterns():
    st.header('Who has higher depression rates?')
    factors = ['sleep_trouble', 'income_group', 'marital_status', 'sex', 'vigorous_activity', 'age_group']
    fig = go.Figure()
    for xi, f in enumerate(factors):
        rate = df.groupby(f, observed=True)['depressed'].mean() * 100
        for yi, (grp, h) in enumerate(rate.items()):
            fig.add_trace(cube(xi, yi, h, RED if h > overall else BLUE))
            fig.add_trace(label3d(xi, yi, h + 1.6, f'{grp}: {h:.1f}%', 10))
    fig.add_trace(plane(-0.5, len(factors) - 0.5, -0.5, 5.5, overall))
    fig.update_layout(scene=dict(xaxis=dict(title='', tickvals=list(range(len(factors))), ticktext=factors),
                                 yaxis=dict(title='group', showticklabels=False), zaxis=dict(title='% depressed'),
                                 camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9))))
    left, right = st.columns([2.4, 1])
    with left:
        show(fig, 600)
        tip()
    st_ = df.groupby('sleep_trouble')['depressed'].mean() * 100
    inc = df.groupby('income_group', observed=True)['depressed'].mean() * 100
    stat(right, f"{st_['Yes']:.1f}% vs {st_['No']:.1f}%", 'trouble sleeping: yes vs no, the biggest gap')
    right.write('')
    stat(right, f"{inc.iloc[0]:.1f}% vs {inc.iloc[-1]:.1f}%", 'below the poverty line vs 4× or more')
    right.write('')
    ag = df.groupby('age_group', observed=True)['depressed'].mean() * 100
    stat(right, f'{ag.min():.1f}–{ag.max():.1f}%', 'age groups: all near the line', NAVY)
    right.caption(f'Red = above the overall {overall:.1f}% (grey plane), blue = below.')


def sleep():
    st.header('Sleep length makes a U-shape')
    s = df[df['sleep_hours'].between(3, 12)].assign(hours=lambda d: d['sleep_hours'].round())
    by_hour = s.groupby('hours')['depressed'].mean() * 100
    grid = s.groupby(['age_group', 'hours'], observed=True)['depressed'].agg(['mean', 'size']).reset_index()
    fig = go.Figure()
    ages = list(df['age_group'].cat.categories)
    for _, r in grid.iterrows():
        h = r['mean'] * 100
        colour = RED if r['hours'] <= 4 or r['hours'] >= 11 else GREEN if 6 <= r['hours'] <= 8 else GREY
        fig.add_trace(cube(r['hours'], ages.index(r['age_group']), h, colour, w=0.7, d=0.5, opacity=0.85))
    # the all-ages U line, drawn behind the bars
    fig.add_trace(go.Scatter3d(x=by_hour.index, y=[-1] * len(by_hour), z=by_hour.values, mode='lines+markers+text',
                               text=[f'{v:.0f}' for v in by_hour.values], textposition='top center',
                               line=dict(color=NAVY, width=6), name='all ages'))
    fig.update_layout(scene=dict(xaxis=dict(title='hours of sleep'),
                                 yaxis=dict(title='', tickvals=[-1] + list(range(4)), ticktext=['all ages'] + ages),
                                 zaxis=dict(title='% depressed'), camera=dict(eye=dict(x=0.2, y=-2.0, z=0.7))))
    show(fig, 560)
    tip()
    a, b, c, d = st.columns(4)
    stat(a, f'{by_hour.loc[3:4].min():.0f}–{by_hour.loc[3:4].max():.0f}%', 'at 3–4 hours of sleep', RED)
    stat(b, f'{by_hour.loc[6:8].min():.0f}–{by_hour.loc[6:8].max():.0f}%', 'at 6–8 hours', GREEN)
    stat(c, f'about {by_hour.loc[11:12].mean():.0f}%', 'at 11–12 hours', RED)
    stat(d, f"r = {df['sleep_hours'].corr(df['phq9_score']):.3f}", 'a straight line can’t fit a U', NAVY)


def corr_figure():
    """3D bars of |Pearson r| for every pair of number columns + PHQ-9 (used on the 'Weak, overlapping signals' slide)."""
    cols = NUMERIC + ['phq9_score']
    corr = df[cols].corr()
    fig = go.Figure()
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                continue
            r = corr.loc[a, b]
            strong = {a, b} == {'income_ratio', 'phq9_score'}
            fig.add_trace(cube(j, i, abs(r), ORANGE if strong else (RED if r > 0 else BLUE), w=0.7, d=0.7))
            if strong or abs(r) >= 0.1:
                fig.add_trace(label3d(j, i, abs(r) + 0.02, f'{r:.2f}', 11))
    fig.update_layout(scene=dict(xaxis=dict(title='', tickvals=list(range(6)), ticktext=cols),
                                 yaxis=dict(title='', tickvals=list(range(6)), ticktext=cols),
                                 zaxis=dict(title='|Pearson r|', range=[0, 1]), camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))))
    return fig, corr


def method():
    st.header('How the models were built')
    cols = st.columns(4)
    for col, (n, title, text) in zip(cols, [
            (1, 'Split first', '80/20, stratified: 4,054 training and 1,014 test adults, both 9.1% depressed.'),
            (2, 'Pipeline', 'Fill gaps, scale numbers, one-hot encode answers → 32 columns. Learned from training data only.'),
            (3, 'Balance', "class_weight = 'balanced': errors on depressed adults count about 10× more."),
            (4, 'Tune', 'Grid search with 5-fold stratified cross-validation, scored by F1.')]):
        col.markdown(f"<div class='card'><div class='big' style='color:{ORANGE}'>{n}</div><h4>{title}</h4>{text}</div>",
                     unsafe_allow_html=True)
    st.write('')
    st.markdown("<div class='dark'><b>Class:</b> depressed (1) or not (0) · "
                "<b>Classifiers:</b> logistic regression, random forest, SVM</div>", unsafe_allow_html=True)


def trap():
    c = counts('LR, not balanced')
    acc = (c['TP'] + c['TN']) / 1014 * 100
    st.markdown(f"<div class='dark'><div class='eyebrow' style='color:{LIGHT_ORANGE}'>The accuracy trap</div>"
                f"<div style='display:flex;gap:80px;margin:20px 0'>"
                f"<div><div class='big' style='color:#F7F6F2;font-size:5rem'>{acc:.1f}%</div>accurate</div>"
                f"<div><div class='big' style='color:{LIGHT_ORANGE};font-size:5rem'>{c['TP']} of 92</div>depressed adults found</div></div>"
                "A model that never predicts depression looks good on accuracy and is useless. So I scored the models on "
                "<b>recall, F1 and ROC-AUC</b>, and used balanced class weights.</div>", unsafe_allow_html=True)


def models_slide():
    st.header('Three classifiers on 1,014 test adults')
    st.caption('80/20 split · 5-fold cross-validation · grid search scored on F1 · balanced class weights · '
               'in cross-validation all three were within 0.01 F1; LR’s 0.316 (CV) vs 0.324 (test) means it is not overfitting')
    rows = []
    for name in ['Logistic regression', 'Random forest', 'SVM', 'LR, not balanced']:
        c = counts(name)
        recall, precision = c['TP'] / 92, c['TP'] / max(c['TP'] + c['FP'], 1)
        f1 = 2 * c['TP'] / (2 * c['TP'] + c['FP'] + c['FN'])
        rows.append({'Model': name, 'Accuracy': (c['TP'] + c['TN']) / 1014, 'Recall': recall, 'F1': f1,
                     'ROC-AUC': roc_auc_score(y_test, score(name)), 'Found': f"{c['TP']} / 92"})
    st.dataframe(pd.DataFrame(rows).set_index('Model').style.format(
        {'Accuracy': '{:.3f}', 'Recall': '{:.3f}', 'F1': '{:.3f}', 'ROC-AUC': '{:.3f}'}), width='stretch')
    left, right = st.columns([2, 1])
    name = right.radio('Confusion matrix of', ['Logistic regression', 'Random forest', 'SVM'])
    c = counts(name)
    fig = go.Figure()
    for (x, y, key, colour, text) in [(0, 0, 'TN', GREY, 'not depressed, cleared'), (1, 0, 'FP', LIGHT_ORANGE, 'false alarm'),
                                       (0, 1, 'FN', RED, 'missed'), (1, 1, 'TP', GREEN, 'found')]:
        fig.add_trace(cube(x, y, c[key], colour, w=0.8, d=0.8))
        fig.add_trace(label3d(x, y, c[key] + 60, f'{text}: {c[key]}', 12))
    fig.update_layout(scene=dict(xaxis=dict(title='guess', tickvals=[0, 1], ticktext=['not depressed', 'depressed']),
                                 yaxis=dict(title='truth', tickvals=[0, 1], ticktext=['not depressed', 'depressed']),
                                 zaxis=dict(title='adults'), camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9))))
    with left:
        show(fig, 460)
        tip()
    right.markdown(f"**Chosen: logistic regression.** Best recall, F1 and AUC, and its odds ratios are easy to explain. "
                   f"The cost: {counts('Logistic regression')['FP']} false alarms.")


def roc():
    st.header('ROC curves: the cut-off is a choice')
    cut = st.slider('Logistic regression cut-off (flag if score ≥ cut-off)', 0.10, 0.90, 0.50, 0.05)
    fig = go.Figure()
    for name, colour in [('Logistic regression', BLUE), ('Random forest', ORANGE), ('SVM', GREEN)]:
        fpr, tpr, _ = roc_curve(y_test, score(name))
        flagged = (tpr * 92 + fpr * 922) / 1014 * 100
        auc = roc_auc_score(y_test, score(name))
        fig.add_trace(go.Scatter3d(x=fpr, y=tpr, z=flagged, mode='lines', line=dict(color=colour, width=6),
                                   name=f'{name} (AUC {auc:.2f})'))
    fig.add_trace(go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 100], mode='lines', line=dict(color='black', dash='dash', width=3),
                               name='guessing (AUC 0.5)'))
    c = counts('Logistic regression', cut)
    x, y = c['FP'] / 922, c['TP'] / 92
    fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[(c['TP'] + c['FP']) / 1014 * 100], mode='markers+text',
                               marker=dict(size=9, color=RED), text=[f'cut-off {cut:.2f}'], name='your cut-off'))
    fig.update_layout(scene=dict(xaxis_title='false positive rate', yaxis_title='true positive rate (recall)',
                                 zaxis_title='% of adults flagged', camera=dict(eye=dict(x=0.1, y=-0.1, z=2.3))))
    left, right = st.columns([2, 1])
    with left:
        show(fig, 560)
        tip()
        st.caption('Seen from the top it is the usual ROC chart; rotate it to see how many adults each point sends for screening.')
    a, b = right.columns(2)
    stat(a, f"{c['TP']} / 92", 'depressed found')
    stat(b, f"{c['FP']}", 'false alarms', NAVY)
    right.write('')
    table = pd.DataFrame([{'cut-off': k, 'found': f"{counts('Logistic regression', k)['TP']} / 92",
                           'false alarms': counts('Logistic regression', k)['FP']} for k in (0.6, 0.5, 0.3)]).set_index('cut-off')
    right.table(table)
    right.markdown(f"**AUC {roc_auc_score(y_test, score('Logistic regression')):.2f}:** a depressed adult gets a higher "
                   'score than a non-depressed one 76% of the time.')


def learned():
    st.header('What the model learned')
    lr = fitted['Logistic regression']
    names = [n.replace('num__', '').replace('cat__', '') for n in lr.named_steps['prep'].get_feature_names_out()]
    w = pd.Series(lr.named_steps['model'].coef_[0], index=names)
    odds = pd.Series({'Trouble sleeping: yes vs no': np.exp(w['sleep_trouble_Yes'] - w['sleep_trouble_No']),
                      'Widowed vs married': np.exp(w['marital_status_Widowed'] - w['marital_status_Married']),
                      'No vigorous activity': np.exp(w['vigorous_activity_No'] - w['vigorous_activity_Yes']),
                      'Told overweight': np.exp(w['told_overweight_Yes'] - w['told_overweight_No']),
                      'Smoked 100+': np.exp(w['smoked_100_Yes'] - w['smoked_100_No']),
                      'Age, per 1 SD older': np.exp(w['age']),
                      'Income, per 1 SD higher': np.exp(w['income_ratio'])})
    rf = fitted['Random forest']
    imp = pd.Series(rf.named_steps['model'].feature_importances_, index=names).sort_values(ascending=False).head(10)
    # 3D bars: one bar per comparison, height = odds ratio (orange raises the odds, blue lowers them)
    fig = go.Figure()
    for xi, (label, v) in enumerate(odds.items()):
        fig.add_trace(cube(xi, 0, v, ORANGE if v > 1 else BLUE, w=0.7, d=0.7))
        fig.add_trace(label3d(xi, 0, v + 0.25, f'{v:.2f}×', 12))
    fig.add_trace(plane(-0.5, len(odds) - 0.5, -0.5, 0.5, 1.0, 'black', 0.18))
    fig.update_layout(scene=dict(xaxis=dict(title='', tickvals=list(range(len(odds))), ticktext=list(odds.index)),
                                 yaxis=dict(title='', showticklabels=False), zaxis=dict(title='odds ratio'),
                                 camera=dict(eye=dict(x=0.4, y=-2.0, z=0.6))))
    left, right = st.columns([1.6, 1])
    with left:
        st.markdown('**Odds ratios (logistic regression)**: the grey plane is 1 = no difference')
        show(fig, 500)
        tip()
    with right:
        st.markdown('**Random forest: top 10 features**')
        f2 = px.bar(imp.sort_values(), orientation='h', text_auto='.3f',
                    color=['sleep trouble' if n.startswith('sleep_trouble') else 'other' for n in imp.sort_values().index],
                    color_discrete_map={'sleep trouble': ORANGE, 'other': BLUE})
        f2.update_layout(showlegend=False, xaxis_title='importance', yaxis_title='')
        show(f2, 430)
        st.markdown(f"<b style='color:{ORANGE}'>Associations, not causes.</b>", unsafe_allow_html=True)


def clusters_slide():
    st.header('k-means: natural groups, without the answer')
    d = df.assign(group=groups)
    rates = d.groupby('group')['depressed'].agg(['size', 'mean'])
    left, right = st.columns([2, 1])
    with left:
        view = st.radio('3D view', [f'all 5 columns (PCA, {pca_share.sum() * 100:.0f}% of the spread)',
                                    'age × income × BMI'], horizontal=True)
        if view.startswith('all'):
            fig = pca_figure(rates)
        else:
            fig = px.scatter_3d(d, x='age', y='income_ratio', z='bmi', color='group', opacity=0.5,
                                color_discrete_map=GROUP_COLOURS, category_orders={'group': list(GROUP_COLOURS)})
            fig.update_traces(marker_size=2.5)
            fig.add_trace(go.Scatter3d(x=centres['age'], y=centres['income_ratio'], z=centres['bmi'], mode='markers+text',
                                       text=[f"{g}: {rates.loc[g, 'mean'] * 100:.1f}%" for g in centres.index],
                                       textposition='top center', marker=dict(size=10, color='black', symbol='diamond'),
                                       name='group centre'))
            fig.update_layout(scene=dict(xaxis_title='age', yaxis_title='income ratio', zaxis_title='BMI'))
        show(fig, 560)
        tip()
    right.caption('5 number columns, scaled 0–1. k-means never saw who was depressed; we checked each group’s rate afterwards.')
    means = d.groupby('group')[['age', 'income_ratio']].mean()          # group averages, as in report Table 10
    table = pd.DataFrame({'Adults': rates['size'], 'Age': means['age'].round(0).astype(int),
                          'Income': means['income_ratio'].round(2), 'Depressed': (rates['mean'] * 100).round(1).astype(str) + '%'})
    right.table(table.loc[list(GROUP_COLOURS)])
    right.markdown('**It rediscovered the income pattern by itself.**')
    e = px.line(x=list(wss), y=list(wss.values()), markers=True, labels={'x': 'k', 'y': 'WSS'})
    e.add_vline(3, line_dash='dash', line_color='red', annotation_text='elbow: k = 3')
    right.markdown('**The elbow chose k = 3** (2→3 groups cuts WSS by 32%, 3→4 by only 14%)')
    with right:
        show(e, 220)


def groupsleep():
    st.header('Sleep trouble, inside every group')
    inside = df.assign(group=groups).groupby(['group', 'sleep_trouble'])['depressed'].mean().unstack() * 100
    inside = inside.loc[list(GROUP_COLOURS)]
    fig = go.Figure()
    for xi, g in enumerate(inside.index):
        for yi, ans in enumerate(['No', 'Yes']):
            h = inside.loc[g, ans]
            fig.add_trace(cube(xi, yi, h, RED if ans == 'Yes' else GREY))
            fig.add_trace(label3d(xi, yi, h + 1.5, f'{h:.1f}%', 13))
    fig.add_trace(plane(-0.5, 2.5, -0.5, 1.5, overall))
    fig.update_layout(scene=dict(xaxis=dict(title='', tickvals=[0, 1, 2], ticktext=list(inside.index)),
                                 yaxis=dict(title='sleep trouble', tickvals=[0, 1], ticktext=['No', 'Yes']),
                                 zaxis=dict(title='% depressed'), camera=dict(eye=dict(x=1.5, y=-1.6, z=0.8))))
    left, right = st.columns([2, 1])
    with left:
        show(fig, 540)
        tip()
    right.caption('k-means could not use sleep trouble (a yes/no answer), so we split each group by it.')
    right.table(inside.round(1).astype(str) + '%')
    lo, hi = inside['No'].min(), inside['Yes'].max()
    right.markdown(f"<b style='color:{ORANGE}'>1 in {100 / hi:.0f} vs 1 in {100 / lo:.0f}:</b> low income plus sleep "
                   f'trouble marks the highest-risk adults ({hi / lo:.0f}× difference).', unsafe_allow_html=True)


def pca_figure(rates):
    """Notebook 21i: the 3 groups on 3 PCA axes (all 5 columns), group centres marked with an X."""
    d = xyz.assign(group=groups)
    fig = px.scatter_3d(d, x='PC1', y='PC2', z='PC3', color='group', opacity=0.45,
                        color_discrete_map=GROUP_COLOURS, category_orders={'group': list(GROUP_COLOURS)})
    fig.update_traces(marker_size=2.5)
    fig.add_trace(go.Scatter3d(x=pca_centres['PC1'], y=pca_centres['PC2'], z=pca_centres['PC3'], mode='markers+text',
                               text=[f"{g}: {rates.loc[g, 'mean'] * 100:.1f}%" for g in pca_centres.index],
                               textposition='top center', marker=dict(size=10, color='black', symbol='x'), name='group centre'))
    fig.update_layout(scene=dict(xaxis_title=f'PC1 ({pca_share[0] * 100:.0f}%)', yaxis_title=f'PC2 ({pca_share[1] * 100:.0f}%)',
                                 zaxis_title=f'PC3 ({pca_share[2] * 100:.0f}%)'))
    return fig


def severity():
    st.header('PHQ-9 scores inside each group')
    order3 = ['Higher income', 'Young, lower income', 'Older, lower income']
    d = df.assign(group=groups)
    left, right = st.columns(2)
    box = px.box(d, x='group', y='phq9_score', color='group', color_discrete_map=GROUP_COLOURS,
                 category_orders={'group': order3}, points='outliers')
    box.add_hline(10, line_dash='dash', line_color='red', annotation_text='cut-off 10')
    means = d.groupby('group')['phq9_score'].mean()
    box.add_trace(go.Scatter(x=order3, y=[means[g] for g in order3], mode='markers+text', text=[f'mean {means[g]:.2f}' for g in order3],
                             textposition='middle right', marker=dict(symbol='diamond', size=11, color=RED), name='mean'))
    box.update_layout(showlegend=False, xaxis_title='', yaxis_title='PHQ-9 score (0–27)')
    with left:
        show(box, 430)
    labels = ['None (0–4)', 'Mild (5–9)', 'Moderate (10–14)', 'Mod. severe (15–19)', 'Severe (20–27)']
    bands = pd.cut(d['phq9_score'], bins=[-1, 4, 9, 14, 19, 27], labels=labels)
    sev = pd.crosstab(d['group'], bands, normalize='index') * 100
    sev.loc['All adults'] = bands.value_counts(normalize=True).reindex(labels) * 100
    long = sev.loc[order3 + ['All adults']].reset_index().melt(id_vars='group', var_name='band', value_name='pct')
    bar = px.bar(long, y='group', x='pct', color='band', orientation='h', text=long['pct'].map(lambda v: f'{v:.0f}%' if v >= 3 else ''),
                 color_discrete_sequence=['#d9d9d9', '#fdd49e', '#fc8d59', '#d7301f', '#7f0000'],
                 category_orders={'band': labels, 'group': order3 + ['All adults']})
    bar.update_layout(xaxis_title='% of the group', yaxis_title='', legend=dict(orientation='h', y=-0.25))
    with right:
        show(bar, 430)
    a, b, c = st.columns(3)
    stat(a, f"{means['Higher income']:.2f} → {means['Older, lower income']:.2f}", 'mean PHQ-9: higher income vs older, lower income')
    stat(b, f"{sev.loc['Higher income', labels[0]]:.0f}% vs {sev.loc['Older, lower income', labels[0]]:.0f}%",
         'score 0–4 (“none”): the whole range shifts, not only past 10')
    stat(c, f"{sev.loc[order3, labels[1]].min():.0f}–{sev.loc[order3, labels[1]].max():.0f}% mild",
         'just under the cut-off: the source of many false alarms', NAVY)


def conclusion():
    st.header('What it means')
    a, b, c = st.columns(3)
    for col, (title, text) in zip([a, b, c], [
            ('Use', 'A GP who hears about sleep problems, money stress or a recent separation has good reason to offer the PHQ-9.'),
            ('Limits', 'A screening score, not a diagnosis · one point in time · sleep trouble overlaps with a PHQ-9 item.'),
            ('Next', 'Lab measures, survey weights, more NHANES cycles, and XGBoost.')]):
        col.markdown(f"<div class='card'><div class='eyebrow'>{title}</div>{text}</div>", unsafe_allow_html=True)
    st.write('')
    st.markdown("<div class='dark' style='font-size:1.6rem'>A simple, explainable model finds <b>two-thirds</b> of people "
                'at risk. For rare outcomes, recall matters more than accuracy.</div>', unsafe_allow_html=True)


def thanks():
    st.markdown(f"<div class='dark' style='padding:60px'><div class='big' style='color:#F7F6F2;font-size:5rem'>Thank you</div>"
                "<h3 style='color:#BFCBE0'>Happy to take questions.</h3>"
                "Kushal Bansal · s8269160 · Data: NHANES 2017–2018 · Code: depression_analysis.ipynb</div>",
                unsafe_allow_html=True)


SLIDES = {'cover': ('Title', cover), 'problem': ('The problem', problem), 'dataset': ('The dataset', dataset),
          'rqs': ('Research questions', rqs), 'explore': ('Exploring the data', explore),
          'boxplots': ('Weak, overlapping signals', boxplots), 'patterns': ('Who has higher rates?', patterns),
          'sleep': ('Sleep U-shape', sleep),
          'method': ('How the models were built', method), 'trap': ('The accuracy trap', trap),
          'models': ('Three classifiers', models_slide), 'roc': ('ROC curves', roc),
          'learned': ('What the model learned', learned), 'clusters': ('k-means', clusters_slide),
          'groupsleep': ('Sleep trouble inside groups', groupsleep),
          'severity': ('PHQ-9 inside groups', severity), 'conclusion': ('What it means', conclusion),
          'thanks': ('Thank you', thanks)}
ORDER = [s for s in NOTES['order'] if s in SLIDES]          # same order as the Slides deck

# ---------- navigation: sidebar list + Back / Next; one number in session_state drives both ----------
if 'nav' not in st.session_state:
    st.session_state.nav = 0


def move(step):
    st.session_state.nav += step


st.sidebar.radio('Slides', range(len(ORDER)), format_func=lambda i: f'{i + 1}. {SLIDES[ORDER[i]][0]}', key='nav')
show_notes = st.sidebar.toggle('Show speaker notes', value=False)

slide_id = ORDER[st.session_state.nav]
SLIDES[slide_id][1]()
if show_notes and NOTES['notes'].get(slide_id):
    st.info(NOTES['notes'][slide_id], icon='🗒️')

st.divider()
back, where, nxt = st.columns([1, 6, 1])
back.button('◀ Back', on_click=move, args=(-1,), disabled=st.session_state.nav == 0, width='stretch')
where.markdown(f"<div style='text-align:center'>{st.session_state.nav + 1} / {len(ORDER)}</div>", unsafe_allow_html=True)
nxt.button('Next ▶', on_click=move, args=(1,), disabled=st.session_state.nav == len(ORDER) - 1, width='stretch')
