# Fashion Insights: A Statistical Analysis of Consumer Preferences and Shopping Behaviour

## College Subject: Statistical Inference Lab
**Academic Domain:** E-Commerce Analytics, Inferential Statistics & Data Science  
**Technologies:** Python 3.10+, Streamlit, SciPy, Statsmodels, Plotly, Scikit-learn, Pandas, NumPy

---

## 📌 Executive Summary & Project Abstract
This project conducts an exhaustive, rigorous statistical inference study on consumer purchasing behavior in the fashion retail industry. Using a verified empirical dataset of 1,000+ consumer transactions, we evaluate consumer spending tendencies across demographic groups, quantify seasonal demand shifts, test hypotheses concerning subscription impacts, and build multivariate predictive models of customer expenditure.

---

## 🎯 Practical Coverage (Aligned with Lab Curriculum)

| # | Topic | Practical Implementation | Statistical Techniques & Tests |
|---|-------|--------------------------|--------------------------------|
| 1 | **Web Scraping & Data Collection** | `scraper.py` & Streamlit UI | HTTP error handling (403), rate-limiting, BeautifulSoup, CSV creation |
| 2 | **Data Cleaning & EDA** | Interactive summary dashboard | Mean, Median, Mode, Variance, IQR, Boxplots, Skewness, Kurtosis |
| 3 | **Probability Distributions & Sampling** | Sampling simulator | Simple Random, Systematic, Stratified Sampling |
| 4 | **Central Limit Theorem (CLT)** | Monte Carlo experiment (k=500-2000) | Sampling distribution of sample means, theoretical vs. empirical SE |
| 5 | **Confidence Intervals & Hypothesis Testing** | Parametric inference engine | One-Sample t-Test & Z-Test, 95% Confidence Intervals, p-values |
| 6 | **Chi-Square Test of Independence** | Categorical contingency analysis | $\chi^2$ test statistic, Cramér's V effect size, p-value |
| 7 | **Covariance & Correlation** | Bivariate & multivariate matrix | Pearson $r$, Spearman $\rho$, Covariance, heatmap |
| 8 | **Correlation Hypothesis Testing** | Significance testing for $r$ | Student's t test for correlation significance ($H_0: \rho = 0$) |
| 9 | **Two-Way ANOVA** | Factorial ANOVA model | Main effects, interaction effect ($F_{AB}$), ANOVA table |
| 10 | **Regression Analysis** | Ordinary Least Squares (OLS) | Multiple linear regression, $R^2$, MAE, RMSE, prediction engine |

---

## 🚀 How to Run Locally

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/<your-username>/fashion-insights-si-lab.git
cd fashion-insights-si-lab

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```
The application will launch automatically at `http://localhost:8501`.

---

## ☁️ Step-by-Step Streamlit Community Cloud Deployment

1. **Push Files to GitHub:**
   Ensure the following 4 files are in your repository root:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `fashion_trends_dataset.csv`

2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **Sign in with GitHub**
   - Click **Create app**

3. **Configure Deployment:**
   - **Repository:** `<your-username>/fashion-insights-si-lab`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - Click **Deploy!**

---

## 🎓 Viva Voce Preparation Guide (Questions & Model Answers)

### Q1: What is the Central Limit Theorem and why is it important in statistical inference?
**Answer:** The Central Limit Theorem (CLT) states that for any independent and identically distributed random sample of size $n$ drawn from a population with finite mean $\mu$ and variance $\sigma^2$, the sampling distribution of the sample mean $\bar{X}$ approaches a normal distribution with mean $\mu$ and standard error $\sigma / \sqrt{n}$ as $n \to \infty$, regardless of the parent distribution's shape. This allows us to use parametric tests (Z-test, t-test) even when the underlying population is moderately skewed, provided sample size $n \ge 30$.

### Q2: What is the difference between a Z-test and a t-test?
**Answer:** A Z-test is used when the population standard deviation $\sigma$ is known or when sample size is very large ($n > 100$). A Student's t-test is used when the population standard deviation $\sigma$ is unknown and must be estimated by the sample standard deviation $s$, with degrees of freedom $df = n - 1$. The t-distribution has fatter tails than the standard normal distribution to account for additional uncertainty.

### Q3: When do we reject the null hypothesis in a Chi-Square test of independence?
**Answer:** We reject $H_0$ if the calculated test statistic $\chi^2 > \chi^2_{\alpha, df}$ or if the resulting $p$-value is less than our significance threshold $\alpha$ (typically $0.05$). Rejection implies that the two categorical variables are statistically dependent and have an association.

### Q4: In Two-Way ANOVA, what does the interaction term measure?
**Answer:** The interaction term ($A \times B$) tests whether the effect of Factor A on the response variable depends on the specific level of Factor B. If the interaction $p$-value $< 0.05$, main effects should not be interpreted in isolation because lines on the interaction plot are not parallel.

### Q5: How do you interpret an $R^2$ of 0.35 in multiple regression?
**Answer:** An $R^2$ of 0.35 means that 35% of the total variance in the response variable (e.g., Purchase Amount) is explained by the linear combination of the independent predictor variables included in the model.
