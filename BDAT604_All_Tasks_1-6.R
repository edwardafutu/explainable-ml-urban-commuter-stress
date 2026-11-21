# =============================================================================
# BDAT 604 - Final Group Project (Tasks 1-6, single run-in-order script)
# =============================================================================
# Members: <Obeng De-Graft Kofi, 22428627> <Edward Afutu Amoni, 22424415> <Samuel Opoku, 22426117>
# Outcome variable : high_stress (binary)
#
# Run this file top to bottom in ONE R session. Order:
#   Task 1  Data exploration and outcome selection
#   Task 2  Regularised (Elastic Net) logistic regression
#   Task 3  Support vector machine (RBF)
#   Task 4  XGBoost ensemble with explainability (SHAP / PDP)
#   Task 5  Big data scaling with sparklyr (+ Random Forest bonus)
#   Task 6  Overall comparison of all models
# Later tasks reuse objects created earlier (stress_train, stress_test, final_result, ...).
# =============================================================================

# The Spark JVM heap must be set BEFORE sparklyr is loaded (used in Task 5).
options(java.parameters = "-Xmx4g")

library(lubridate)
library(tidyverse)
library(tidymodels)
library(glmnet)
library(kernlab)
library(xgboost)
library(vip)
library(DALEXtra)
library(shapviz)
library(sparklyr)
library(knitr)

tidymodels_prefer()  # resolve function-name clashes in favour of tidymodels

# =============================================================================
# TASK 1: Data Exploration and Outcome Selection
# =============================================================================

# -----------------------------------------------------------------------------
# 0. DATA GENERATING FUNCTION 
# -----------------------------------------------------------------------------

simulate_numcws <- function(n = 500000, seed = 604) {
  library(tidyverse)
  set.seed(seed)
  
  # ── 1. IDENTIFIERS ──────────────────────────────────────────────────────────
  
  respondent_id  <- 1:n
  household_id   <- sample(1:(n * 0.4), n, replace = TRUE)
  
  country <- sample(
    c("Ghana", "Nigeria", "Kenya", "Ethiopia",
      "Senegal", "South Africa"),
    n, TRUE,
    prob = c(0.20, 0.25, 0.18, 0.12, 0.10, 0.15)
  )
  
  city <- case_when(
    country == "Ghana"        ~ sample(c("Accra", "Kumasi"), n, TRUE,
                                       prob = c(0.70, 0.30)),
    country == "Nigeria"      ~ sample(c("Lagos", "Abuja", "Kano"), n, TRUE,
                                       prob = c(0.55, 0.30, 0.15)),
    country == "Kenya"        ~ sample(c("Nairobi", "Mombasa"), n, TRUE,
                                       prob = c(0.75, 0.25)),
    country == "Ethiopia"     ~ sample(c("Addis Ababa"), n, TRUE),
    country == "Senegal"      ~ sample(c("Dakar"), n, TRUE),
    country == "South Africa" ~ sample(c("Johannesburg", "Cape Town"), n, TRUE,
                                       prob = c(0.60, 0.40)),
    TRUE ~ "Unknown"
  )
  
  city_tier <- case_when(
    city %in% c("Lagos", "Nairobi", "Johannesburg",
                "Cape Town", "Accra") ~ "Tier1",
    city %in% c("Abuja", "Addis Ababa",
                "Dakar", "Kumasi")    ~ "Tier2",
    TRUE                              ~ "Tier3"
  )
  
  survey_year <- sample(2022:2024, n, TRUE,
                        prob = c(0.30, 0.35, 0.35))
  
  survey_month <- sample(1:12, n, TRUE)
  
  # ── 2. DEMOGRAPHICS ─────────────────────────────────────────────────────────
  
  age <- pmax(16, pmin(75, round(rnorm(n, mean = 34, sd = 11))))
  
  sex <- sample(c("Male", "Female"), n, TRUE,
                prob = c(0.52, 0.48))
  
  education <- sample(
    c("None", "Primary", "Secondary",
      "Tertiary", "Postgraduate"),
    n, TRUE,
    prob = c(0.06, 0.18, 0.32, 0.33, 0.11)
  )
  
  marital_status <- sample(
    c("Single", "Married", "Divorced", "Widowed"),
    n, TRUE,
    prob = c(0.38, 0.48, 0.09, 0.05)
  )
  
  household_size <- pmax(1, round(rnorm(n, mean = 4.2, sd = 1.8)))
  
  num_children <- pmax(0, rpois(n, lambda = ifelse(
    marital_status == "Married", 2.1, 0.8
  )))
  
  # ── 3. EMPLOYMENT & INCOME ──────────────────────────────────────────────────
  
  employment_type <- sample(
    c("Formal", "Informal", "Self-employed",
      "Unemployed", "Student"),
    n, TRUE,
    prob = c(0.32, 0.28, 0.22, 0.10, 0.08)
  )
  
  sector <- case_when(
    employment_type %in% c("Formal", "Informal") ~
      sample(c("Public", "Private", "NGO"), n, TRUE,
             prob = c(0.30, 0.62, 0.08)),
    employment_type == "Self-employed" ~ "Private",
    TRUE ~ "None"
  )
  
  # Monthly income in USD (PPP-adjusted)
  base_income <- case_when(
    education == "None"          ~  180,
    education == "Primary"       ~  280,
    education == "Secondary"     ~  420,
    education == "Tertiary"      ~  780,
    education == "Postgraduate"  ~ 1400,
    TRUE ~ 420
  )
  
  monthly_income <- round(
    base_income *
      ifelse(sex == "Male", 1.15, 1.0) *
      ifelse(city_tier == "Tier1", 1.35,
             ifelse(city_tier == "Tier2", 1.10, 1.0)) *
      case_when(
        employment_type == "Formal"       ~ 1.20,
        employment_type == "Informal"     ~ 0.75,
        employment_type == "Self-employed"~ 1.00,
        employment_type == "Unemployed"   ~ 0.15,
        employment_type == "Student"      ~ 0.25,
        TRUE ~ 1.0
      ) *
      rlnorm(n, 0, 0.35),
    2
  )
  
  # ── 4. TRANSPORT BEHAVIOUR ──────────────────────────────────────────────────
  
  primary_mode <- sample(
    c("Bus", "Minibus/Trotro", "Motorcycle taxi",
      "Private car", "Walking", "Train/Metro"),
    n, TRUE,
    prob = c(0.22, 0.28, 0.18, 0.14, 0.12, 0.06)
  )
  
  has_private_vehicle <- rbinom(n, 1, prob = plogis(
    -2.5 +
      0.04 * (monthly_income / 100) -
      0.5  * (employment_type == "Unemployed") +
      0.4  * (city_tier == "Tier1")
  ))
  
  num_transfers <- pmax(0, rpois(n, lambda = case_when(
    primary_mode == "Bus"             ~ 1.8,
    primary_mode == "Minibus/Trotro"  ~ 1.2,
    primary_mode == "Motorcycle taxi" ~ 0.4,
    primary_mode == "Private car"     ~ 0.1,
    primary_mode == "Walking"         ~ 0.0,
    primary_mode == "Train/Metro"     ~ 0.9,
    TRUE ~ 1.0
  )))
  
  # One-way commute time (minutes)
  commute_time_mins <- pmax(
    5,
    round(
      rnorm(
        n,
        mean = case_when(
          city_tier == "Tier1" ~ 65,
          city_tier == "Tier2" ~ 48,
          TRUE                 ~ 35
        ) +
          8 * num_transfers +
          ifelse(primary_mode == "Walking", -20, 0) +
          ifelse(primary_mode == "Private car", -10, 0),
        sd = 22
      )
    )
  )
  
  # Monthly transport cost (USD)
  base_cost <- case_when(
    primary_mode == "Bus"             ~ 18,
    primary_mode == "Minibus/Trotro"  ~ 25,
    primary_mode == "Motorcycle taxi" ~ 35,
    primary_mode == "Private car"     ~ 80,
    primary_mode == "Walking"         ~  2,
    primary_mode == "Train/Metro"     ~ 22,
    TRUE ~ 20
  )
  
  monthly_transport_cost <- round(
    base_cost *
      (1 + 0.008 * commute_time_mins) *
      (1 + 0.3 * num_transfers) *
      ifelse(city_tier == "Tier1", 1.25, 1.0) *
      rlnorm(n, 0, 0.25),
    2
  )
  
  # Transport cost as share of income
  transport_cost_pct <- monthly_transport_cost /
    pmax(monthly_income, 50)
  
  # ── 5. ROAD & SAFETY CONDITIONS ─────────────────────────────────────────────
  
  road_quality_score <- pmin(10, pmax(1, round(
    rnorm(
      n,
      mean = case_when(
        city_tier == "Tier1" ~ 6.2,
        city_tier == "Tier2" ~ 5.0,
        TRUE                 ~ 3.8
      ),
      sd = 1.5
    ),
    1
  )))
  
  experienced_accident <- rbinom(n, 1, prob = plogis(
    -3.5 +
      0.4  * (primary_mode == "Motorcycle taxi") +
      0.3  * (primary_mode == "Minibus/Trotro") -
      0.08 * road_quality_score +
      0.015 * commute_time_mins +
      0.2  * (city_tier == "Tier3")
  ))
  
  # ── 6. HEALTH & WELLBEING ───────────────────────────────────────────────────
  
  bmi <- pmax(16, pmin(45, rnorm(
    n,
    mean = 24.5 + 0.06 * (age - 25) +
      ifelse(sex == "Female", 0.8, 0),
    sd = 4.2
  )))
  
  physical_activity_days <- pmax(0, pmin(7, round(
    7 - 0.04 * commute_time_mins +
      2.0 * (primary_mode == "Walking") +
      rnorm(n, 0, 1.5)
  )))
  
  # Annual health expenditure (USD)
  annual_health_expenditure <- round(
    pmax(0,
         50 +
           1.2 * age +
           30  * (bmi > 30) +
           80  * experienced_accident +
           20  * (physical_activity_days < 2) -
           15  * (physical_activity_days > 4) +
           0.05 * monthly_income +
           rlnorm(n, 0, 0.5) * 40
    ),
    2
  )
  
  # Stress level (latent variable used to generate outcomes)
  stress_latent <- (
    0.03  * commute_time_mins +
      0.5   * num_transfers -
      0.1   * road_quality_score +
      0.8   * experienced_accident -
      0.002 * monthly_income +
      0.4   * (employment_type == "Unemployed") -
      0.3   * (primary_mode == "Private car") +
      0.2   * (primary_mode == "Walking") * (-1) +
      rnorm(n, 0, 1.0)
  )
  
  # Binary: high stress (top ~35%)
  high_stress <- as.integer(stress_latent >
                              quantile(stress_latent, 0.65))
  
  # Commute satisfaction (1-5 Likert, ordered)
  commute_satisfaction <- pmin(5, pmax(1, round(
    5.5 -
      0.025 * commute_time_mins -
      0.4   * num_transfers +
      0.15  * road_quality_score -
      0.5   * high_stress +
      0.3   * (primary_mode == "Private car") +
      0.2   * (primary_mode == "Walking") +
      rnorm(n, 0, 0.8)
  )))
  
  # ── 7. JOB & ACCESSIBILITY ──────────────────────────────────────────────────
  
  years_at_job <- pmax(0, round(
    rlnorm(n, meanlog = 1.2, sdlog = 0.9)
  ))
  
  job_accessibility_score <- pmin(100, pmax(0, round(
    60 -
      0.15 * commute_time_mins +
      5.0  * (city_tier == "Tier1") +
      3.0  * (city_tier == "Tier2") +
      0.01 * monthly_income +
      2.0  * (primary_mode %in% c("Train/Metro", "Bus")) -
      3.0  * (employment_type == "Unemployed") +
      rnorm(n, 0, 10),
    1
  )))
  
  # ── 8. DIGITAL & INFRASTRUCTURE ─────────────────────────────────────────────
  
  has_smartphone    <- rbinom(n, 1, prob = plogis(
    -1.0 + 0.03 * (monthly_income / 100) +
      0.5 * (education %in% c("Tertiary", "Postgraduate")) +
      0.4 * (city_tier == "Tier1")
  ))
  
  uses_ride_app     <- rbinom(n, 1, prob = ifelse(
    has_smartphone == 1,
    plogis(-1.5 + 0.02 * (monthly_income / 100) +
             0.6 * (city_tier == "Tier1")),
    0
  ))
  
  internet_access   <- rbinom(n, 1, prob = plogis(
    -0.5 + 0.8 * has_smartphone +
      0.3 * (city_tier == "Tier1")
  ))
  
  # ── 9. BEHAVIOURAL OUTCOMES ─────────────────────────────────────────────────
  
  # Intent to switch primary transport mode
  mode_switch_intent <- rbinom(n, 1, prob = plogis(
    -1.0 +
      0.03  * commute_time_mins +
      0.4   * num_transfers -
      0.15  * commute_satisfaction +
      0.5   * high_stress -
      0.3   * (primary_mode == "Private car") +
      0.2   * uses_ride_app +
      rnorm(n, 0, 0.3)
  ))
  
  # Transport poverty: spending > 25% of income on transport
  transport_poverty <- as.integer(transport_cost_pct > 0.25)
  
  # Tenure risk / job loss in past 12 months
  job_loss_12m <- rbinom(n, 1, prob = plogis(
    -3.0 +
      0.5  * high_stress +
      0.4  * (employment_type == "Informal") +
      0.8  * (employment_type == "Unemployed") -
      0.02 * years_at_job +
      0.3  * (commute_time_mins > 90) +
      rnorm(n, 0, 0.3)
  ))
  
  # ── 10. SURVEY METADATA ─────────────────────────────────────────────────────
  
  survey_date <- as.Date("2022-01-01") +
    days(sample(0:1094, n, TRUE))
  
  interview_mode <- sample(
    c("Face-to-face", "Phone", "Online"),
    n, TRUE,
    prob = c(0.45, 0.35, 0.20)
  )
  
  # ── 11. ASSEMBLE DATA FRAME ──────────────────────────────────────────────────
  
  data.frame(
    # Identifiers
    respondent_id,
    household_id,
    
    # Geography & time
    country,
    city,
    city_tier,
    survey_year,
    survey_month,
    survey_date,
    interview_mode,
    
    # Demographics
    age,
    sex,
    education,
    marital_status,
    household_size,
    num_children,
    
    # Employment & income
    employment_type,
    sector,
    monthly_income,
    years_at_job,
    
    # Transport behaviour
    primary_mode,
    has_private_vehicle,
    num_transfers,
    commute_time_mins,
    monthly_transport_cost,
    transport_cost_pct,
    
    # Road & safety
    road_quality_score,
    experienced_accident,
    
    # Health & wellbeing
    bmi,
    physical_activity_days,
    annual_health_expenditure,
    stress_latent,
    
    # Job & accessibility
    job_accessibility_score,
    
    # Digital
    has_smartphone,
    uses_ride_app,
    internet_access,
    
    # Outcomes
    commute_satisfaction,
    high_stress,
    mode_switch_intent,
    transport_poverty,
    job_loss_12m,
    
    stringsAsFactors = FALSE
  )
}

# -----------------------------------------------------------------------------
# 1. GENERATING THE THE FULL SIMULATED POPULATION
# -----------------------------------------------------------------------------

urban_population <- simulate_numcws(n = 500000, seed = 757)    
#rationale: required for task 5

glimpse(urban_population)

# -----------------------------------------------------------------------------
# 1.1 OUTCOME AND PREDICTOR SELECTION
# -----------------------------------------------------------------------------

# Outcome  : high_stress   (binary, 0/1)
# predictors (10, within the 5-15 allowed range). T
#   commute_time_mins, num_transfers, road_quality_score,
#   experienced_accident, monthly_income, employment_type, primary_mode,
#   city_tier, age, sex
outcome_var    <- "high_stress"
predictor_vars <- c("commute_time_mins", "num_transfers", "road_quality_score",
                    "experienced_accident", "monthly_income", "employment_type",
                    "primary_mode", "city_tier", "age", "sex")

analysis_vars <- c(outcome_var, predictor_vars, "country")

# -----------------------------------------------------------------------------
# 1.2 Sampling and EXPLORATORY DATA ANALYSIS
# -----------------------------------------------------------------------------

#===== STRATIFIED SAMPLING ===========
# Rationale for choosing STRATIFICATION (by country)
#country has deliberately unequal population shares (Nigeria 25%, Ghana
#20%, Kenya 18%, South Africa 15%, Ethiopia 12%, Senegal 10%).  stratifying
# GUARANTEES every country -- including the smallest, Senegal and
# Ethiopia -- is proportionally represented in the analysis sample, which
# matters because road quality, accident risk and income (all stress
# drivers) vary materially by country.

# We use PROPORTIONAL stratified sampling by country: every stratum is
# sampled at the same rate, so the sample's country composition mirrors the
# population's composition almost exactly.

TARGET_N <- 10000                              # within the 5,000-40,000 range
sampling_frac <- TARGET_N / nrow(urban_population)

set.seed(575)   # separate seed for the sampling stage - deliberately
# different from group level seed so the sampling draw is not
# mechanically correlated with the variable-generation draws

urban_data <- urban_population %>%
  group_by(country) %>%                        # stratify by country
  mutate(u = runif(n())) %>%                    # random uniform per stratum
  filter(u <= sampling_frac) %>%                # keep same rate in each stratum
  ungroup() %>%
  select(-u) %>%
  select(respondent_id, all_of(analysis_vars))

nrow(urban_data)   # should be close to TARGET_N (~10,000)

# Check the sample reproduces the population's country composition
bind_rows(
  urban_population %>% count(country) %>% mutate(pct = 100*n/sum(n), source = "Population"),
  urban_data  %>% count(country) %>% mutate(pct = 100*n/sum(n), source = "Sample")
) %>%
  select(source, country, pct) %>%
  pivot_wider(names_from = source, values_from = pct) %>%
  arrange(desc(Population))


#======== EXPLORATORY DATA ANALYSIS ===============
urban_data <- urban_data %>%
  mutate(high_stress = factor(high_stress, levels = c(0,1),
                              labels = c("Not high stress", "High stress")))

## a.Class balance check (categorical outcome) ---------------------------
class_balance <- urban_data  %>%
  count(high_stress) %>%
  mutate(pct = 100 * n / sum(n))
print(class_balance)

# "Figure 1. Distribution and Class Balance Check of High Stress" 
ggplot(class_balance, aes(x = high_stress, y = n, fill = high_stress)) +
  geom_col() +
  geom_text(aes(label = paste0(n, "\n(", round(pct,1), "%)")), vjust = -0.3) +
  scale_fill_manual(values = c("steelblue", "indianred"), guide = "none") +
  labs(title = "",
       x = NULL, y = "Count") +
  theme_minimal()
ggsave("fig1_class_balance.png", width = 5, height = 4.3, dpi = 150)

##b.Summary statistics for predictors, stratified by outcome class ------
# ── Define predictor groups ──────────────────────────────────────────────────
continuous_vars <- c("commute_time_mins", "num_transfers", 
                     "road_quality_score", "monthly_income", "age")

categorical_vars <- c("experienced_accident", "employment_type", 
                      "primary_mode", "city_tier", "sex")

# ── Table 1: Continuous predictors, stratified by outcome ────────────────────
continuous_summary <- urban_data %>%
  select(high_stress, all_of(continuous_vars)) %>%
  pivot_longer(cols = all_of(continuous_vars),
               names_to = "Predictor", values_to = "Value") %>%
  group_by(high_stress, Predictor) %>%
  summarise(
    Mean   = round(mean(Value, na.rm = TRUE), 2),
    SD     = round(sd(Value, na.rm = TRUE), 2),
    Median = round(median(Value, na.rm = TRUE), 2),
    IQR    = paste0(round(quantile(Value, 0.25, na.rm = TRUE), 1), " - ",
                    round(quantile(Value, 0.75, na.rm = TRUE), 1)),
    Min    = round(min(Value, na.rm = TRUE), 1),
    Max    = round(max(Value, na.rm = TRUE), 1),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from  = high_stress,
    values_from = c(Mean, SD, Median, IQR, Min, Max)
  )

print(continuous_summary)


# ── Table 2: Categorical / binary predictors, stratified by outcome ──────────
categorical_summary <- urban_data %>%
  select(high_stress, all_of(categorical_vars)) %>%
  mutate(
    experienced_accident = ifelse(experienced_accident == 1, "Yes", "No"),
    across(-experienced_accident, as.character)
  ) %>%
  pivot_longer(cols = all_of(categorical_vars),
               names_to = "Predictor", values_to = "Category") %>%
  count(high_stress, Predictor, Category) %>%
  group_by(high_stress, Predictor) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  ungroup() %>%
  mutate(Label = paste0(n, " (", pct, "%)")) %>%
  select(-n, -pct) %>%
  pivot_wider(names_from = high_stress, values_from = Label)

print(categorical_summary)

## c.Bivariate plots -------------------------------------------------------

# (i) Continuous predictor vs. outcome: commute_time_mins
#Figure 2. Commute Time by Stress Class
ggplot(urban_data, aes(x = high_stress, y = commute_time_mins, fill = high_stress)) +
  geom_boxplot(alpha = 0.6) +
  scale_fill_manual(values = c("steelblue", "indianred"), guide = "none") +
  labs(title = "",
       x = NULL, y = "Commute time (minutes)") +
  theme_minimal()
ggsave("fig2_boxplot_stress.png", width = 5, height = 4, dpi = 150)

# (ii) Categorical predictor vs. outcome: primary_mode (stacked composition)
#Figure 3. Share Reporting High Stress, by Primary Mode
ggplot(urban_data, aes(x = primary_mode, fill = high_stress)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("steelblue", "indianred")) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "",
       x = "Primary mode", y = "% of respondents", fill = NULL) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave("fig3_stackedbar_mode.png", width = 7, height = 4, dpi = 150)








# =============================================================================
# TASK 2: Regularised Classification of high_stress
# Model family : penalised (regularised) LOGISTIC REGRESSION via glmnet
# Reason       : high_stress is a binary outcome, so logistic regression is the
#                natural GLM; a regularisation penalty is added (Task 2 brief) to
#                control variance, handle correlated commute-related predictors,
#                and perform automatic variable selection.
# =============================================================================

tidymodels_prefer()          # resolve function-name clashes in favour of tidymodels

# -----------------------------------------------------------------------------
# 2.0 BUILD THE MODELLING DATA FRAME
# -----------------------------------------------------------------------------
# We reuse the stratified analysis sample (urban_data, ~10,000 rows) and the
# 10 predictors selected and justified in Task 1.1.
#
# Two deliberate encoding choices:
#  (a) The outcome is re-labelled so that "High_stress" is the SECOND factor
#      level. glmnet / logistic regression model the probability of the second
#      level, so this makes every fitted coefficient read directly as the effect
#      on the log-odds of BEING high-stress (positive = raises stress risk).
#  (b) experienced_accident (0/1) is turned into a labelled factor ("No"/"Yes")
#      so the recipe treats it as a category (a clean Yes-vs-No dummy) rather
#      than scaling a raw 0/1 column.

model_data <- urban_data %>%
  select(all_of(c(outcome_var, predictor_vars))) %>%
  mutate(
    high_stress = fct_recode(high_stress,
                             Not_high    = "Not high stress",
                             High_stress = "High stress"),
    experienced_accident = factor(experienced_accident, levels = c(0, 1),
                                  labels = c("No", "Yes")),
    across(where(is.character), as.factor)          # employment_type, primary_mode, city_tier, sex
  )

levels(model_data$high_stress)   # confirm: "Not_high" then "High_stress" (event = 2nd level)
glimpse(model_data)

# -----------------------------------------------------------------------------
# 2.0.1 TRAIN / TEST SPLIT  (do this BEFORE any preprocessing -> no data leakage)
# -----------------------------------------------------------------------------
# 75/25 split, STRATIFIED by high_stress so both sets keep the ~35% / 65% class
# balance found in Task 1 (important because the outcome is imbalanced).

set.seed(757)                                        # group-level seed (reproducibility)
stress_split <- initial_split(model_data, prop = 0.75, strata = high_stress)
stress_train <- training(stress_split)
stress_test  <- testing(stress_split)

cat("Training rows:", nrow(stress_train), "\n")
cat("Test rows:    ", nrow(stress_test),  "\n")

# Check the stratification held
bind_rows(
  stress_train %>% count(high_stress) %>% mutate(pct = 100*n/sum(n), set = "Train"),
  stress_test  %>% count(high_stress) %>% mutate(pct = 100*n/sum(n), set = "Test")
)

# -----------------------------------------------------------------------------
# 2.1 PREPROCESSING RECIPE                                              [2.1]
# -----------------------------------------------------------------------------
# Every step below is required so the regularisation penalty is applied FAIRLY.
#   step_log       : monthly_income is strongly right-skewed (lognormal in the
#                    NUMCWS design; confirmed by the skew seen in Task 1 EDA).
#                    A log10 transform pulls in the long income tail so a few
#                    high earners do not dominate the fit. The other continuous
#                    predictors (commute_time_mins, num_transfers,
#                    road_quality_score, age) are roughly symmetric and are left
#                    untransformed.
#   step_dummy     : one-hot encode the categorical predictors (employment_type,
#                    primary_mode, city_tier, sex, experienced_accident) into
#                    0/1 indicator columns that glmnet can use.
#   step_normalize : centre and scale ALL numeric predictors to mean 0, sd 1.
#                    This is ESSENTIAL for regularisation: the L1/L2 penalty is
#                    scale-sensitive, so without normalisation a predictor
#                    measured on a large scale (e.g. income) would be penalised
#                    differently from one on a small scale (e.g. num_transfers).
#   step_nzv       : drop any near-zero-variance dummy (a category so rare it
#                    carries almost no information and can destabilise the fit).

stress_recipe <- recipe(high_stress ~ ., data = stress_train) %>%
  step_log(monthly_income, base = 10) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_nzv(all_predictors())

# Inspect what the recipe will do, and preview the processed training matrix
summary(stress_recipe)
prep(stress_recipe) %>% bake(new_data = NULL) %>% glimpse()

# -----------------------------------------------------------------------------
# 2.2 REGULARISATION CHOICE AND JUSTIFICATION                          [2.2]
# -----------------------------------------------------------------------------
# CHOICE: ELASTIC NET (we tune the mixture alpha, so the model is free to sit
#         anywhere on the Ridge (alpha = 0) <-> Lasso (alpha = 1) continuum and
#         let cross-validation pick the best point).
#
# WHY, with reference to the three required points:
#  * Predictors vs sample size: after dummy encoding we have ~18 predictor
#    columns against ~7,400 training rows (n >> p). Over-fitting risk is
#    therefore modest, but a penalty still stabilises the estimates and guards
#    against noise dummies.
#  * Expected sparsity: we hand-picked 10 theoretically relevant predictors in
#    Task 1, so we do NOT expect heavy sparsity - but weak dummies (e.g. some
#    sex / city_tier / employment levels) may add little. The Lasso component of
#    Elastic Net can shrink those toward (or to) zero for a simpler model.
#  * Correlated predictors: several stress drivers are correlated
#    (commute_time_mins, num_transfers, city_tier and monthly_income all move
#    together). Pure Lasso tends to keep one of a correlated group and drop the
#    others arbitrarily; the Ridge component of Elastic Net shares the weight
#    across correlated predictors (grouping effect), giving more stable, more
#    interpretable coefficients. Elastic Net gives us BOTH selection and
#    stability, so we tune alpha rather than fixing it in advance.

enet_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

enet_wf <- workflow() %>%
  add_recipe(stress_recipe) %>%
  add_model(enet_spec)

# -----------------------------------------------------------------------------
# 2.3 HYPERPARAMETER TUNING                                            [2.3]
# -----------------------------------------------------------------------------
# 10-fold cross-validation, STRATIFIED by high_stress (keeps the class balance
# in every fold - the brief asks for stratified CV where appropriate, and the
# outcome is imbalanced). We search a regular grid over:
#   penalty (lambda) : 30 values, log-spaced from 1e-4 to 1e0
#   mixture (alpha)  : 6 values, 0.0, 0.2, ..., 1.0  (Ridge -> Lasso)
# Selection metric = ROC AUC: it is threshold-independent and robust to class
# imbalance, so it is the honest way to rank models. event_level = "second"
# keeps High_stress as the positive class.

set.seed(757)
stress_folds <- vfold_cv(stress_train, v = 10, strata = high_stress)

enet_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  mixture(range = c(0, 1)),
  levels = c(30, 6)
)

enet_tune <- tune_grid(
  enet_wf,
  resamples = stress_folds,
  grid      = enet_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(event_level = "second")
)

# Optimal hyperparameter values + cross-validated AUC (show_best reports both)
show_best(enet_tune, metric = "roc_auc", n = 5)
best_enet <- select_best(enet_tune, metric = "roc_auc")
best_enet

# --- Regularisation path plot (required by 2.3) ---
# Fit the model at the chosen (lambda, alpha) so we can read glmnet's full
# coefficient path, then plot how each coefficient shrinks as lambda grows.
final_enet_fit <- enet_wf %>%
  finalize_workflow(best_enet) %>%
  fit(data = stress_train)

# Figure 4. Regularisation path - coefficient size vs lambda (chosen lambda dashed)
coef_path <- final_enet_fit %>%
  extract_fit_engine() %>%
  broom::tidy() %>%
  filter(term != "(Intercept)")

ggplot(coef_path, aes(log10(lambda), estimate, colour = term)) +
  geom_line(show.legend = FALSE) +
  geom_vline(xintercept = log10(best_enet$penalty), linetype = "dashed") +
  labs(x = expression(log[10](lambda)), y = "Standardised coefficient") +
  theme_minimal()
ggsave("fig4_reg_path.png", width = 7, height = 4.3, dpi = 150)

# -----------------------------------------------------------------------------
# 2.4 FINAL MODEL AND INTERPRETATION                                   [2.4]
# -----------------------------------------------------------------------------

# STEP 1 - Fit the final model on the FULL training set at the chosen
# (lambda, alpha) and evaluate ONCE on the locked test set. last_fit() does both
# in a single step (as taught in Lecture 1). event_level = "second" keeps
# High_stress as the positive class for the metrics.
final_result <- enet_wf %>%
  finalize_workflow(best_enet) %>%
  last_fit(stress_split,
           metrics = metric_set(roc_auc, f_meas),
           control = control_last_fit(event_level = "second"))

# STEP 2 - Test-set performance, reported on two metrics (AUC and F1)
collect_metrics(final_result)

# STEP 3 - Interpret the coefficients. exp(coefficient) = ODDS RATIO; because the
# predictors were normalised, each odds ratio is the multiplicative change in the
# odds of high stress for a one-SD increase (or Yes vs No for a dummy). Sorted so
# the strongest drivers appear first.
enet_coefs <- final_result %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  mutate(odds_ratio = exp(estimate)) %>%
  arrange(desc(abs(estimate)))
print(enet_coefs, n = Inf)

# STEP 4 - Which predictors did the Lasso penalty shrink EXACTLY to zero?
enet_coefs %>% filter(term != "(Intercept)", estimate == 0)

# -----------------------------------------------------------------------------
# WRITTEN INTERPRETATION  (Task 2.4 - drivers of high commuter stress)
# -----------------------------------------------------------------------------
# Cross-validation selected mixture alpha = 1 (the Lasso end of the Elastic Net
# continuum) with penalty lambda ~= 0.0045. The model generalises well: test
# AUC ~= 0.905 and F1 ~= 0.75 are essentially identical to the cross-validated
# figures (CV AUC ~= 0.90, F1 ~= 0.75), so there is no sign of over-fitting.
#
# Interpreting the strongest retained predictors in the NUMCWS context (odds
# ratios for a one-SD change, holding the others fixed):
#   * monthly_income  (OR ~ 0.26): the strongest signal. Higher-income commuters
#     have far LOWER odds of high stress - a one-SD rise in (log) income cuts the
#     odds by roughly three-quarters. Wealthier respondents can afford faster,
#     more comfortable travel and shorter commutes.
#   * commute_time_mins (OR ~ 3.7): the strongest RISK factor. A one-SD longer
#     commute multiplies the odds of high stress almost four-fold - long journeys
#     are the dominant stressor, exactly the lever transport policy can act on.
#   * num_transfers (OR ~ 2.2): each extra mode change (bus/trotro hops) roughly
#     doubles the odds of high stress; smoother, better-integrated routes would
#     reduce transfers and stress.
#   * road_quality_score (OR ~ 0.77): better perceived road quality LOWERS stress
#     odds - infrastructure investment plausibly improves commuter wellbeing.
#   * experienced_accident = Yes (OR ~ 1.27): commuters who had a road accident in
#     the past year have ~27% higher odds of high stress, a road-safety signal.
# Among the surviving mode dummies, private-car (OR ~ 0.85) and walking
# (OR ~ 0.88) commuters have modestly lower stress odds than the bus baseline.
#
# On sparsity (Lasso variable selection): the L1 penalty shrank SEVEN dummy
# coefficients EXACTLY to zero - age, sex (Male), city_tier (Tier2 and Tier3),
# self-employment, and the minibus/trotro and motorcycle-taxi mode indicators.
# This implies those predictors add no independent information about high stress
# once commute burden, income and road safety are in the model (e.g. city-tier
# effects are already captured through commute time and income). The result is a
# more parsimonious, easier-to-explain model that keeps only the genuine drivers
# - a useful steer for the policy model in later tasks.



# =============================================================================
# TASK 3: Support Vector Machine (SVM) classification of high_stress
# =============================================================================
        

# -----------------------------------------------------------------------------
# 3.1 KERNEL CHOICE AND JUSTIFICATION                                   
# -----------------------------------------------------------------------------

set.seed(757)
svm_train <- stress_train %>% slice_sample(n = 2000)   # subsample for feasibility

# SVMs REQUIRE normalised inputs so we reuse the Task 2 recipe (log income,
# dummy-encode, normalise, drop near-zero-variance), so preprocessing is identical.
svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_wf <- workflow() %>%
  add_recipe(stress_recipe) %>%
  add_model(svm_spec)

# -----------------------------------------------------------------------------
# 3.2 HYPERPARAMETER TUNING                                             
# -----------------------------------------------------------------------------
# Tune cost C (margin softness) and rbf_sigma (gamma, kernel width) with 5-fold
# cross-validation, ranked by ROC AUC (same metric as Task 2).
set.seed(757)
svm_folds <- vfold_cv(svm_train, v = 5, strata = high_stress)

svm_grid <- grid_regular(
  cost(range      = c(-1, 3)),
  rbf_sigma(range = c(-4, 0)),
  levels = 4                       
)

svm_tune <- svm_wf %>%
  tune_grid(
    resamples = svm_folds,
    grid      = svm_grid,
    metrics   = metric_set(roc_auc),
    control   = control_grid(event_level = "second")
  )

# Optimal hyperparameter values + cross-validated AUC
show_best(svm_tune, metric = "roc_auc", n = 5)
best_svm <- select_best(svm_tune, metric = "roc_auc")
best_svm

# Figure 5. Tuning surface: CV AUC across cost and rbf_sigma
autoplot(svm_tune, metric = "roc_auc") +
  labs(x = "Cost (C)", y = "ROC AUC (5-fold CV)", colour = "rbf_sigma", title = "") +
  theme_minimal()
ggsave("fig5_svm_tuning.png", width = 7, height = 4.3, dpi = 150)

# -----------------------------------------------------------------------------
# 3.3 FINAL MODEL EVALUATION                                            
# -----------------------------------------------------------------------------
# Fit the final SVM on the subsample at the chosen (C, sigma), then evaluate on
# the SAME full test set as Task 2 using the SAME metrics (AUC and F1).
final_svm <- svm_wf %>%
  finalize_workflow(best_svm) %>%
  fit(data = svm_train)

svm_pred <- final_svm %>%
  predict(new_data = stress_test, type = "prob") %>%
  bind_cols(predict(final_svm, new_data = stress_test)) %>%
  bind_cols(stress_test %>% select(high_stress))

# Test-set metrics
roc_auc(svm_pred, truth = high_stress, .pred_High_stress, event_level = "second")
f_meas(svm_pred, truth = high_stress, estimate = .pred_class, event_level = "second")
conf_mat(svm_pred, truth = high_stress, estimate = .pred_class)

# Number of support vectors (model complexity)
svm_engine <- extract_fit_engine(final_svm)
nSV(svm_engine)                              # count of support vectors
nSV(svm_engine) / nrow(svm_train)            # fraction of training points retained


# -----------------------------------------------------------------------------
# 3.4 COMPARISON WITH TASK 2                                            
# -----------------------------------------------------------------------------
task3_comparison <- tibble(
  Model = c("Elastic Net logistic (Task 2)", "SVM - RBF (Task 3)"),
  `Test AUC` = round(c(
    collect_metrics(final_result) %>% filter(.metric == "roc_auc") %>% pull(.estimate),
    roc_auc(svm_pred, high_stress, .pred_High_stress, event_level = "second")$.estimate), 3),
  `Test F1` = round(c(
    collect_metrics(final_result) %>% filter(.metric == "f_meas") %>% pull(.estimate),
    f_meas(svm_pred, high_stress, estimate = .pred_class, event_level = "second")$.estimate), 3),
  Interpretability = c("High - odds ratio per predictor", "Low - kernel black box"),
  `Training cost`  = c("Low - glmnet on full 7,443 rows", "High - kernlab on 2,000-row subsample")
)
print(task3_comparison)


# =============================================================================
# TASK 4: Ensemble Method (XGBoost) with Explainability
# =============================================================================


# -----------------------------------------------------------------------------
# 4.0 REUSE THE TASK 2 TRAIN/TEST SPLIT AND BUILD THE ENSEMBLE RECIPE
# -----------------------------------------------------------------------------

cat("Training rows:", nrow(stress_train), " | Test rows:", nrow(stress_test), "\n")

# Same preprocessing recipe as Task 2 (log-transform income, dummy-encode
# categoricals, normalise, drop near-zero-variance columns). Trees do not
# NEED normalisation, but -- as Lecture 3 notes -- it does not hurt them,
# and reusing stress_recipe keeps the exact same feature matrix across
# Tasks 2-4 for a fair comparison in Task 6.

ens_recipe <- stress_recipe   # identical recipe object from Task 2

# -----------------------------------------------------------------------------
# 4.2 HYPERPARAMETER TUNING                                              [4.2]
# -----------------------------------------------------------------------------
# We tune SEVEN hyperparameters via (trees, tree_depth, learn_rate, mtry, min_n,
# loss_reduction, sample_size). 5-fold STRATIFIED CV, Latin-hypercube search 
# over 25 combinations,stratified by high_stress

# scale_pos_weight from the TRAINING data class counts
train_counts <- table(stress_train$high_stress)
spw <- as.numeric(train_counts["Not_high"] / train_counts["High_stress"])
cat("scale_pos_weight =", round(spw, 3), "\n")

xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  mtry           = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune()
) %>%
  set_engine("xgboost", scale_pos_weight = spw) %>%
  set_mode("classification")

xgb_wf <- workflow() %>%
  add_recipe(ens_recipe) %>%
  add_model(xgb_spec)

set.seed(757)
stress_folds5 <- vfold_cv(stress_train, v = 5, strata = high_stress)

# Random / Latin-hypercube search (more efficient than a full grid over 7 hyperparameters)
xgb_grid <- grid_space_filling(
  trees(range            = c(200, 1000)),
  tree_depth(range       = c(3, 8)),
  learn_rate(range       = c(-3, -1)),          # log10 scale: 0.001 to 0.1
  mtry(range             = c(5L, 14L)),   # integer count of predictors,
  min_n(range            = c(5, 30)),
  loss_reduction(range   = c(-5, 0)),
  sample_size            = sample_prop(range = c(0.5, 1.0)),
  size = 25                                     # >= 20 combinations required
)  

xgb_tune <- xgb_wf %>%
  tune_grid(
    resamples = stress_folds5,
    grid      = xgb_grid,
    metrics   = metric_set(roc_auc, f_meas),
    control   = control_grid(verbose = TRUE, save_pred = TRUE,
                             event_level = "second")
  )

# Best hyperparameter combination and its cross-validated performance
show_best(xgb_tune, metric = "roc_auc", n = 5)
best_xgb <- select_best(xgb_tune, metric = "roc_auc")
best_xgb

# -----------------------------------------------------------------------------
# FIT THE FINAL MODEL AND EVALUATE ONCE ON THE LOCKED TEST SET
# -----------------------------------------------------------------------------
final_xgb <- xgb_wf %>%
  finalize_workflow(best_xgb) %>%
  last_fit(stress_split,
           metrics = metric_set(roc_auc, f_meas),
           control = control_last_fit(event_level = "second"))

collect_metrics(final_xgb)   

# -----------------------------------------------------------------------------
# 4.3 GLOBAL EXPLAINABILITY                                              
# -----------------------------------------------------------------------------

xgb_engine <- final_xgb %>% extract_fit_engine()

## (a) Variable importance plot (top 10), with written interpretation ------
# Figure 4. XGBoost Variable Importance for High Stress 
final_xgb %>%
  extract_fit_parsnip() %>%
  vip(num_features = 10, geom = "col",
      aesthetics = list(fill = "steelblue", alpha = 0.85)) +
  labs(title = "",
       subtitle = "Gain-based importance -- top 10 predictors",
       x = "Predictor", y = "Relative importance (gain)") +
  theme_minimal() + coord_flip()
ggsave("fig4_vip.png", width = 7, height = 4.5, dpi = 150)


## (b) Partial dependence plots for >= 2 predictors -------------------------
# Build a DALEX explainer (model-agnostic wrapper)
explainer_xgb <- explain_tidymodels(
  model   = final_xgb %>% extract_workflow(),
  data    = stress_test %>% select(-high_stress),
  y       = as.numeric(stress_test$high_stress == "High_stress"),
  label   = "XGBoost High-Stress Model",
  verbose = FALSE
)

# PDP 1: commute_time_mins
#Figure 5a. PDP: Effect of Commute Time on P(High Stress)
pdp_commute <- model_profile(explainer_xgb, variables = "commute_time_mins",
                             type = "partial")
plot(pdp_commute) +
  labs(title = "",
       x = "Commute time (minutes)", y = "Average predicted P(High stress)") +
  theme_minimal()
ggsave("fig5a_pdp_commute.png", width = 6, height = 4, dpi = 150)


# PDP 2: monthly_income (log-scale, as entered into the recipe)
#Figure 5b. PDP: Effect of Monthly Income on P(High Stress)
pdp_income <- model_profile(explainer_xgb, variables = "monthly_income",
                            type = "partial")
plot(pdp_income) +
  labs(title = "",
       x = "Monthly income (USD)", y = "Average predicted P(High stress)") +
  theme_minimal()
ggsave("fig5b_pdp_income.png", width = 6, height = 4, dpi = 150)


## (c) SHAP beeswarm (summary) plot -----------------------------------------
# Prepare the baked test matrix (TreeSHAP needs a numeric matrix, Lecture 4)
rec_prepped <- final_xgb %>% extract_recipe()

# Bake ONCE, keep as a data frame
baked_test <- bake(rec_prepped, new_data = stress_test %>% select(-high_stress))

# Matrix version for the actual TreeSHAP computation
test_matrix <- baked_test %>% as.matrix()

shap_values <- shapviz(
  object = xgb_engine,
  X_pred = test_matrix,
  X      = baked_test        # <- same encoded columns, so names line up
)

## Sanity check: does the raw booster predict P(High_stress) or P(Not_high)?
## Compare its raw prediction to the workflow's correctly-labelled probability
## (the workflow's predict() respects event_level = "second", the raw
## xgb.Booster does not -- see tidymodels/parsnip issue #796).

wf_prob_high <- predict(final_xgb %>% extract_workflow(),
                        stress_test, type = "prob")$.pred_High_stress

raw_prob <- predict(xgb_engine, test_matrix)   # raw booster's own P(label = 1)

cat("Correlation (should be ~ -1 if flipped, ~ +1 if fine):",
    cor(wf_prob_high, raw_prob), "\n")

## -> if the correlation comes back strongly NEGATIVE, the raw booster's
##    "positive class" is Not_high, not High_stress, so we flip the SHAP
##    decomposition's sign before plotting/interpreting it.

if (cor(wf_prob_high, raw_prob) < 0) {
  shap_values$S        <- -shap_values$S
  shap_values$baseline <- -shap_values$baseline
}

# Key outputs
dim(shap_values$S) # n_test x p matrix of SHAP values
head(shap_values$S) # each row sums to: prediction-base_value

# Figure 6. SHAP beeswarm -- global feature impact across all test observations
#Figure 6. SHAP Beeswarm: Global Feature Impact on High Stress
sv_importance(shap_values, kind = "beeswarm", max_display = 10) +
  labs(title = "",
       subtitle = "Each dot = one respondent. Colour = feature value (yellow = high, blue = low).")
ggsave("fig6_shap_beeswarm.png", width = 7.5, height = 5, dpi = 150)


#-----------------------------------------------------------------------------
# 4.4 LOCAL EXPLAINABILITY                                               [4.4]
# -----------------------------------------------------------------------------
# Select three test-set observations: (1) TYPICAL -- correctly classified as
# High_stress, with a predicted probability close to the median among correct
# High_stress predictions; (2) BORDERLINE -- predicted probability closest to
# the 0.5 cutoff; (3) SURPRISING -- the most CONFIDENTLY WRONG prediction
# (largest gap between predicted probability and the true label).

test_preds <- collect_predictions(final_xgb) %>%
  arrange(.row) %>%                              # <-- FIX 1: restore original
  #     stress_test row order
  #     before bind_cols(),
  #     so row_id lines up
  #     with baked_test /
  #     shap_values indexing
  bind_cols(stress_test %>% select(commute_time_mins, num_transfers,
                                   road_quality_score, experienced_accident,
                                   monthly_income, employment_type,
                                   primary_mode)) %>%
  mutate(row_id = row_number())

# (1) Typical: correct High_stress prediction, near the median predicted prob
correct_hs <- test_preds %>% filter(.pred_class == "High_stress",
                                    high_stress == "High_stress")
typical_row <- correct_hs %>%
  mutate(d = abs(.pred_High_stress - median(.pred_High_stress))) %>%
  slice_min(d, n = 1, with_ties = FALSE) %>%      # <-- FIX 2: break ties,
  pull(row_id)                                    #     force a single row

# (2) Borderline: predicted probability closest to 0.5
borderline_row <- test_preds %>%
  mutate(d = abs(.pred_High_stress - 0.5)) %>%
  slice_min(d, n = 1, with_ties = FALSE) %>%      # <-- FIX 2
  pull(row_id)

# (3) Surprising: most confidently WRONG prediction
surprising_row <- test_preds %>%
  filter(.pred_class != high_stress) %>%
  mutate(conf = ifelse(.pred_class == "High_stress",
                       .pred_High_stress, 1 - .pred_High_stress)) %>%
  slice_max(conf, n = 1, with_ties = FALSE) %>%   # <-- FIX 2
  pull(row_id)

cat("Typical row:", typical_row, " | Borderline row:", borderline_row,
    " | Surprising row:", surprising_row, "\n")


# SHAP waterfall plots for the three cases (Figure 7a-c)
for (case in list(list(id = typical_row,    name = "typical"),
                  list(id = borderline_row, name = "borderline"),
                  list(id = surprising_row, name = "surprising"))) {
  sv_waterfall(shap_values, row_id = case$id, max_display = 10) +
    labs(title = paste0("Figure 7. SHAP Waterfall - ", case$name, " case (row ", case$id, ")")) +
    theme_minimal()
  ggsave(paste0("fig7_waterfall_", case$name, ".png"), width = 7, height = 4.5, dpi = 150)
}


# -----------------------------------------------------------------------------
# WRITTEN INTERPRETATION -- LOCAL EXPLAINABILITY                      [4.4]
# -----------------------------------------------------------------------------
# Pull this respondent's actual (raw, unstandardized) monthly income for the write-up
stress_test$monthly_income[1474]

# Or, to see it alongside the other raw values for context:
stress_test[1474, c("monthly_income", "commute_time_mins", "num_transfers",
                    "employment_type", "experienced_accident", "primary_mode",
                    "high_stress")]

# "This prediction was made primarily because the respondent's monthly 
# income($1701) is high relative to other commuters, which strongly lowered the 
# model's estimated stress risk. Their number of commute transfers(0) played a 
# smaller role, also nudging the prediction toward lower stress. Despite this, 
# the respondent actually reported high stress, showing that income alone doesn't 
# guarantee low stress for every commuter."




# =============================================================================
# TASK 5: Big Data Scaling with sparklyr
# =============================================================================

# --------------------------------------------------------------------------------------
# 5.0 BIG DATA SCALING (same 10 predictors as Task 2)
# --------------------------------------------------------------------------------------

# Connect to local Spark (simulates a cluster on your machine)
config <- spark_config()
config$`sparklyr.shell.driver-memory` <- "4g"
config$spark.driver.memory            <- "4g"
config$spark.executor.memory          <- "4g"
sc <- spark_connect(master = "local[*]", config = config)

# Copy NUMCWS dataset to Spark
numcws_spark <- copy_to(sc, urban_population, "numcws_data",
                        overwrite = TRUE)

# Train-test split in Spark (same seed as Task 2 for consistency)
numcws_spark_split <- numcws_spark %>%
  sdf_random_split(train = 0.75, test = 0.25, seed = 757)

numcws_train_spark <- numcws_spark_split$train
numcws_test_spark  <- numcws_spark_split$test

# -----------------preprocessing: log-transform income, same as Task 2 recipe-----------
numcws_train_spark <- numcws_train_spark %>%
  mutate(log_income = log(monthly_income + 1))

numcws_test_spark <- numcws_test_spark %>%
  mutate(log_income = log(monthly_income + 1))

# ---------------LOGISTIC REGRESSION: Task 2's 10-predictor set--------------------------
# categorical predictors (employment_type, primary_mode, city_tier, sex) are
# one-hot encoded automatically by Spark's RFormula interface inside ml_logistic_regression
spark_logit <- ml_logistic_regression(
  numcws_train_spark,
  high_stress ~ age +
    log_income +
    commute_time_mins +
    num_transfers +
    road_quality_score +
    experienced_accident +
    employment_type +
    primary_mode +
    city_tier +
    sex
)

predictions <- ml_predict(
  spark_logit,
  numcws_test_spark
)

predictions %>%
  group_by(high_stress, prediction) %>%
  summarise(n = n()) %>%
  collect()

summary(spark_logit)

# ---------------Confusion matrix and metrics---------------------------------------------
cm <- predictions %>%
  group_by(high_stress, prediction) %>%
  summarise(n = n(), .groups = "drop") %>%
  collect()

TN <- cm$n[cm$high_stress == 0 & cm$prediction == 0]
FP <- cm$n[cm$high_stress == 0 & cm$prediction == 1]
FN <- cm$n[cm$high_stress == 1 & cm$prediction == 0]
TP <- cm$n[cm$high_stress == 1 & cm$prediction == 1]

accuracy <- (TP + TN) / (TP + TN + FP + FN)
accuracy

auc <- ml_binary_classification_evaluator(
  predictions,
  label_col = "high_stress",
  raw_prediction_col = "rawPrediction",
  metric_name = "areaUnderROC"
)
auc

precision  <- TP / (TP + FP)
recall     <- TP / (TP + FN)
f1         <- 2 * (precision * recall) / (precision + recall)
specificity <- TN / (TN + FP)

precision
recall
f1
specificity



# --------------------------------------------------------------------------------------
#  BONUS: Spark Random Forest (same 10 predictors) as Task 4 (stress_train / stress_test)
# --------------------------------------------------------------------------------------

# Fit Spark RF on the Task 2-4 training set (same rows XGBoost was trained on)
task4_train_for_spark <- stress_train %>%
  mutate(
    high_stress           = ifelse(high_stress == "High_stress", 1, 0),
    experienced_accident  = ifelse(experienced_accident == "Yes", 1, 0),
    log_income            = log(monthly_income + 1)
  ) %>%
  select(high_stress, age, log_income, commute_time_mins, num_transfers,
         road_quality_score, experienced_accident, employment_type,
         primary_mode, city_tier, sex)

task4_test_for_spark <- stress_test %>%
  mutate(
    high_stress           = ifelse(high_stress == "High_stress", 1, 0),
    experienced_accident  = ifelse(experienced_accident == "Yes", 1, 0),
    log_income            = log(monthly_income + 1)
  ) %>%
  select(high_stress, age, log_income, commute_time_mins, num_transfers,
         road_quality_score, experienced_accident, employment_type,
         primary_mode, city_tier, sex)

task4_train_spark <- copy_to(sc, task4_train_for_spark, "task4_train_data", overwrite = TRUE)
task4_test_spark  <- copy_to(sc, task4_test_for_spark,  "task4_test_data",  overwrite = TRUE)

# Fit Spark RF on the Task 2-4 training set (same rows XGBoost was trained on)
spark_rf_task4 <- ml_random_forest(
  task4_train_spark,
  high_stress ~ age + log_income + commute_time_mins + num_transfers +
    road_quality_score + experienced_accident + employment_type +
    primary_mode + city_tier + sex,
  type = "classification",
  num_trees = 300,
  max_depth = 6,
  feature_subset_strategy = "sqrt",
  seed = 757
)

rf_task4_predictions <- ml_predict(spark_rf_task4, task4_test_spark)

# ---- metrics ----
rf_task4_cm <- rf_task4_predictions %>%
  group_by(high_stress, prediction) %>%
  summarise(n = n(), .groups = "drop") %>%
  tidyr::complete(high_stress, prediction, fill = list(n = 0)) %>%
  collect()

TN <- rf_task4_cm$n[rf_task4_cm$high_stress == 0 & rf_task4_cm$prediction == 0]
FP <- rf_task4_cm$n[rf_task4_cm$high_stress == 0 & rf_task4_cm$prediction == 1]
FN <- rf_task4_cm$n[rf_task4_cm$high_stress == 1 & rf_task4_cm$prediction == 0]
TP <- rf_task4_cm$n[rf_task4_cm$high_stress == 1 & rf_task4_cm$prediction == 1]

accuracy_rf_task4  <- (TP + TN) / (TP + TN + FP + FN)
precision_rf_task4 <- TP / (TP + FP)
recall_rf_task4    <- TP / (TP + FN)
f1_rf_task4        <- 2 * precision_rf_task4 * recall_rf_task4 / (precision_rf_task4 + recall_rf_task4)
specificity_rf_task4 <- TN / (TN + FP)

auc_rf_task4 <- ml_binary_classification_evaluator(
  rf_task4_predictions,
  label_col = "high_stress",
  raw_prediction_col = "rawPrediction",
  metric_name = "areaUnderROC"
)

accuracy_rf_task4
auc_rf_task4
precision_rf_task4
recall_rf_task4
f1_rf_task4
specificity_rf_task4

# Variable importance, same call as before
rf_task4_importance <- ml_feature_importances(spark_rf_task4)
rf_task4_importance


# =============================================================================
# TASK 6: Overall Comparison and Conclusion
# =============================================================================

# ---- 1. Collect the test-set metrics produced in Tasks 2-5 -------------------
get_metric <- function(res, m) collect_metrics(res) %>% filter(.metric == m) %>% pull(.estimate)

# Task 2: Elastic Net (last_fit result)
test_auc <- get_metric(final_result, "roc_auc")
test_f1  <- get_metric(final_result, "f_meas")

# Task 3: SVM
svm_auc <- roc_auc(svm_pred, high_stress, .pred_High_stress, event_level = "second")$.estimate
svm_f1  <- f_meas(svm_pred, high_stress, estimate = .pred_class, event_level = "second")$.estimate
n_sv    <- kernlab::nSV(svm_engine)

# Task 4: XGBoost (last_fit result)
xgb_test_auc <- get_metric(final_xgb, "roc_auc")
xgb_test_f1  <- get_metric(final_xgb, "f_meas")

# Task 5: Spark logistic regression and Spark Random Forest
spark_auc    <- auc          # from the Spark logistic regression block
spark_f1     <- f1
rf_auc_task4 <- auc_rf_task4
rf_f1_task4  <- f1_rf_task4

# ---- 2. Sizes and search-grid counts ------------------------------------------
n_train_small <- nrow(stress_train)
n_test_small  <- nrow(stress_test)
n_train_spark <- sdf_nrow(numcws_train_spark)
n_test_spark  <- sdf_nrow(numcws_test_spark)
n_enet_combos <- nrow(enet_grid)
n_xgb_combos  <- nrow(xgb_grid)
n_svm_train   <- nrow(svm_train)

auc_all <- c(test_auc, svm_auc, xgb_test_auc, spark_auc, rf_auc_task4)
f1_all  <- c(test_f1,  svm_f1,  xgb_test_f1,  spark_f1,  rf_f1_task4)
auc_gap_pp <- 100 * (max(auc_all) - min(auc_all))

# ---- 3. Comparative performance table (Table 11) ----------------------------
task6_table <- tibble(
  Model = c("Elastic Net logistic (Task 2)", "SVM - RBF (Task 3)",
            "XGBoost (Task 4)", "Spark logistic regression (Task 5)",
            "Spark Random Forest (Task 5, bonus)"),
  Type = c("Regularised linear", "Kernel (non-linear)", "Gradient-boosted trees",
           "Regularised linear, distributed", "Bagged trees, distributed"),
  AUC = round(auc_all, 3),
  F1  = round(f1_all, 3),
  `Approx. training cost` = c(
    paste0("Low - ", n_enet_combos, "-combination 10-fold CV grid on ~",
           scales::comma(round(n_train_small, -2)), " rows"),
    paste0("High relative to data size - subsampled to ",
           scales::comma(n_svm_train), " rows for the 5-fold CV grid"),
    paste0("Moderate - ", n_xgb_combos, "-point Latin-hypercube search, 5-fold CV on ~",
           scales::comma(round(n_train_small, -2)), " rows"),
    paste0("Low per fit, needs a live Spark session - one unpenalised fit on ",
           scales::comma(n_train_spark), " rows"),
    paste0("Low - 300 trees, depth 6, single fit (no CV) on ~",
           scales::comma(round(n_train_small, -2)), " rows")
  ),
  Interpretability = c(
    "High - coefficients read directly as odds ratios",
    paste0("Low - no coefficients; ", n_sv, " support vectors"),
    "Moderate - black-box, but SHAP/PDP recover global and local interpretability",
    "High - same interpretable form as Task 2",
    "Moderate - impurity-based importances, no per-observation SHAP"
  )
)

print(kable(task6_table,
            caption = "Table 11: Comparative performance of all models, Tasks 2-5"))
write_csv(task6_table, "task6_comparison_table.csv")

# ---- 4. Summary statements (all numbers computed) ---------------------------
cat(sprintf(
  "\nAll five models land within %.1f percentage points of one another on AUC (%.3f-%.3f),\ndespite training sizes from ~%s to ~%s rows.\n",
  auc_gap_pp, min(auc_all), max(auc_all),
  scales::comma(round(n_train_small, -2)),
  scales::comma(round(n_train_spark, -3))))

cat(sprintf(
  "\nRecommendation: deploy the Elastic Net logistic regression (AUC = %.3f, F1 = %.3f).\nXGBoost reached AUC = %.3f but F1 = %.3f.\n",
  test_auc, test_f1, xgb_test_auc, xgb_test_f1))

# ---- Close the Spark connection -----------------------------------------------
spark_disconnect(sc)

