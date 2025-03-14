#Probing data analysis clean

setwd("~/OneDrive/Coding/AISC/SAE_error_probing_AMI")

if (!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if (!require(magrittr)) install.packages("magrittr"); library(magrittr)

library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(skimr)
library(modelsummary)
library(fixest)
library(readr)
myTheme <- theme(plot.title = element_text(size = 14),
                 panel.background = element_rect(fill = '#F2F2ED'),
                 axis.title = element_text(size = 10),
                 axis.text = element_text(size = 10, colour = 'black'),
                 legend.title = element_text(size = 12),
                 legend.position = "right",
                 legend.background = element_rect(linetype = 3,size = 0.5, color = 'black', fill = 'grey94'),
                 legend.text = element_text(size = 10),
                 legend.key = element_rect(size = 0.5, linetype = 1, color = 'black'))

#I also have some nice colors that I use in my various graphs.
nicepurp <- "#A88DBF"
niceblue <- '#38A5E0'
nicegreen <- '#A3DCC0'

custom_colors <- c("#2ECC71", "#A3E635", "#F4D03F", "#F39C12", "#E74C3C", "#C0392B")

#Functions
#####



probe_summarizer <- function(probe_name, some_probes, save_fig = TRUE){
  # Renaming the columns to replace spaces with underscores
  colnames(some_probes) <- gsub(" ", "_", colnames(some_probes))
  
  # Modified probe_summary code
  probe_summary <- some_probes %>% group_by(`Feature_Type`) %>% 
    summarize(mean_test_accuracy = mean(`Test_Accuracy`), 
              se_test_accuracy = sd(`Test_Accuracy`)/sqrt(n()),
              mean_test_loss = mean(`Test_Loss`), 
              se_test_loss = sd(`Test_Loss`)/sqrt(n()),
              mean_test_ROC = mean(Test_ROC_AUC),
              se_test_ROC = sd(Test_ROC_AUC)/sqrt(n())) %>% ungroup()
  
  
  probe_summary %<>% mutate(graph_labels = c(
    "SAE Error + Latents",
    "SAE Latents",
    "SAE Error", 
    "Residual"))
  
  
  ggplot(probe_summary, aes(x = graph_labels, 
                            y = mean_test_accuracy,)) +
    geom_point() + myTheme + 
    labs(y = 'Probe Out of Sample Accuracy', x = NULL,
         title = str_c(probe_name, " Out of Sample Accuracy"),
         subtitle = "Error bars indicate randomness from using different seeds") + 
    geom_errorbar(aes(ymin = mean_test_accuracy - 1.96*se_test_accuracy, 
                      ymax = mean_test_accuracy + 1.96*se_test_accuracy),
                  width = 0.1)+
    scale_y_continuous(labels = scales::percent)
  if(save_fig){
    ggsave(str_c("reports/figures/", probe_name, "_oos_accuracy.png"), width = 6, height = 4, scale = 1.2)
  }
  
  
  ggplot(probe_summary, aes(x = graph_labels, 
                            y = mean_test_loss,)) +
    geom_point() + myTheme + 
    labs(y = 'Probe Out of Sample Loss', x = NULL,
         title = str_c(probe_name, " Out of Sample Loss"),
         subtitle = "Error bars indicate randomness from using different seeds") + 
    geom_errorbar(aes(ymin = mean_test_loss - 1.96*se_test_loss, 
                      ymax = mean_test_loss + 1.96*se_test_loss),
                  width = 0.1)+
    scale_y_continuous()
  if(save_fig){
    ggsave(str_c("reports/figures/", probe_name, "_oos_loss.png"), width = 6, height = 4, scale = 1.2)
  }
    
  print(modelsummary(feols(Test_Loss ~ Feature_Type | as.factor(Seed), data = some_probes, vcov = ~Seed),
                     fmt = "%.3f",        # 3 decimal places
                     stars = TRUE,title = str_c(probe_name,": Test loss mean differences Layer 19"),           # Show confidence intervals instead of std errors
                     statistic = c("conf.int"),
                     conf_level = 0.95,
                     coef_map = c(
                       "Feature_Typesae_input" = "Residual - SAE Error",
                       "Feature_Typesae_recons" = "SAE Reconstruction - SAE Error"
                     ),
                     gof_map = list(
                       list(raw = "nobs", clean = "Num.Obs.", fmt = 0),
                       list(raw = "r.squared", clean = "R²", fmt = "%.3f"),
                       list(raw = "std.error", clean = "Std.Errors", fmt = "%.3f")
                     ),# Add significance stars
                     output = "markdown"))
  
  print(modelsummary(feols(Test_Accuracy ~ Feature_Type | as.factor(Seed), data = some_probes, vcov = ~Seed),
                     fmt = "%.3f",        # 3 decimal places
                     stars = TRUE,title =str_c(probe_name,": Test accuracy mean differences Layer 19"),          # Show confidence intervals instead of std errors
                     statistic = c("conf.int"),
                     conf_level = 0.95,
                     coef_map = c(
                       "Feature_Typesae_input" = "Residual - SAE Error",
                       "Feature_Typesae_recons" = "SAE Reconstruction - SAE Error"
                     ),
                     gof_map = list(
                       list(raw = "nobs", clean = "Num.Obs.", fmt = 0),
                       list(raw = "r.squared", clean = "R²", fmt = "%.3f"),
                       list(raw = "std.error", clean = "Std.Errors", fmt = "%.3f")
                     ),# Add significance stars
                     output = "markdown"))

  print(modelsummary(feols(Test_ROC_AUC ~ Feature_Type | as.factor(Seed), data = some_probes, vcov = ~Seed),
                    fmt = "%.3f",        # 3 decimal places
                    stars = TRUE,title =str_c(probe_name,": Test ROC AUC mean differences Layer 19"),          # Show confidence intervals instead of std errors
                    statistic = c("conf.int"),
                    conf_level = 0.95,
                    coef_map = c(
                       "Feature_Typesae_input" = "Residual - SAE Error",
                       "Feature_Typesae_recons" = "SAE Reconstruction - SAE Error"
                     ),
                     gof_map = list(
                       list(raw = "nobs", clean = "Num.Obs.", fmt = 0),
                       list(raw = "r.squared", clean = "R²", fmt = "%.3f"),
                       list(raw = "std.error", clean = "Std.Errors", fmt = "%.3f")
                     ),# Add significance stars
                     output = "markdown"))
  return(probe_summary)
}


#Read in logistic regression results


headline_probes <- read_csv("data/processed/logistic_combined_gemma_2_2b_layer_19_width_16k_canonical_k100_again/probe_results_headline_fp.csv")
headline_summary <- probe_summarizer('Probing for Headline (Front Page)', headline_probes, save_fig = FALSE)

headline_probes %>% group_by(Feature_Type) %>% skim()

manhattan_probes <- read_csv("data/processed/logistic_combined_gemma_2_2b_layer_19_width_16k_canonical_k100/probe_results_man_borough.csv")
manhattan_summary <- probe_summarizer('Probing for in Manhattan', manhattan_probes, save_fig = FALSE)

manhattan_probes %>% group_by(Feature_Type) %>% skim()

tw_happy <- read_csv("data/processed/logistic_combined_gemma_2_2b_layer_19_width_16k_canonical_k100/probe_results_twt_happy.csv")
tw_happy_summary <- probe_summarizer('Probing for Happiness in Tweets', tw_happy, save_fig = FALSE)

tw_happy %>% select(-starts_with("Cosine")) %>%group_by(Feature_Type) %>% skim()

combined_plot <- headline_summary %>% mutate(Dataset = "Frontpage headlines") %>%
  add_row(manhattan_summary %>% mutate(Dataset = "Location in Manhattan")) %>%
  add_row(tw_happy_summary %>% mutate(Dataset = "Happiness in Tweet"))

ggplot(combined_plot, aes(x = graph_labels, 
                          y = mean_test_ROC,
                          color = Feature_Type,
                          shape = Feature_Type)) +  # Add color aesthetic
  geom_point() + 
  myTheme + 
  labs(y = 'Probe Out of Sample ROC AUC', 
       x = NULL,
       title = "Out of sample ROC AUC with logistic regression",
       subtitle = "Results on gemma 2b. Error bars indicate randomness from using different seeds",
       color = "Probe Location", shape = "Probe Location") + 
  geom_errorbar(aes(ymin = mean_test_ROC - 1.96*se_test_ROC, 
                    ymax = mean_test_ROC + 1.96*se_test_ROC),
                width = 0.1) +
  scale_y_continuous() + 
  facet_wrap(~Dataset, scales = 'free') +
  theme(axis.text.x = element_blank(),  # Remove x-axis text
        axis.ticks.x = element_blank(),
        legend.position = 'bottom') +  # Remove x-axis ticks
  scale_color_manual(values = c(nicepurp, niceblue, nicegreen,"#F39C12"))
ggsave("reports/figures/k100v1.png", width = 6, height = 4, scale = 1.6, dpi = 400)


manhattan_probes <- read_csv('results/logistic_gemma_2_2b_layer_19_width_16k_canonical/probe_results_man_borough.csv')
manhattan_summary <- probe_summarizer('Probing for in Manhattan', manhattan_probes, save_fig = FALSE)


truth_probes <- read_csv('results/logistic_meta_llama_Llama_3.1_8B_l19r_8x/probe_results_truth.csv')
truth_summary <- probe_summarizer('Probing for Truth in Cities Dataset', truth_probes, save_fig = FALSE)

headline_probes <- read_csv('results/logistic_meta_llama_Llama_3.1_8B_l19r_8x/probe_results_headline_fp.csv')
headline_summary <- probe_summarizer('Probing for Headline (Front Page)', headline_probes, save_fig = FALSE)

manhattan_probes <- read_csv('results/logistic_meta_llama_Llama_3.1_8B_l19r_8x/probe_results_man_borough.csv')
manhattan_summary <- probe_summarizer('Probing for in Manhattan', manhattan_probes, save_fig = FALSE)

tw_happy <- read_csv("results/logistic_meta_llama_Llama_3.1_8B_l19r_8x/probe_results_twt_happy.csv")
tw_happy_summary <- probe_summarizer('Probing for Happiness in Tweets', tw_happy, save_fig = FALSE)

basketball <- read_csv("results/logistic_meta_llama_Llama_3.1_8B_l19r_8x/probe_results_ath_basketball.csv")
basketball_summary <- probe_summarizer('Probing for Basketball Atheletes', basketball, save_fig = FALSE)

combined_plot <- truth_summary %>% mutate(Dataset = "Truth") %>% 
  add_row(headline_summary %>% mutate(Dataset = "Frontpage headlines")) %>%
  add_row(manhattan_summary %>% mutate(Dataset = "Location in Manhattan")) %>%
  add_row(tw_happy_summary %>% mutate(Dataset = "Happiness in Tweet")) %>%
  add_row(basketball_summary %>% mutate(Dataset = "Athelete plays basketball"))

ggplot(combined_plot, aes(x = graph_labels, 
                          y = mean_test_accuracy,
                          color = graph_labels,
                          shape = graph_labels)) +  # Add color aesthetic
  geom_point() + 
  myTheme + 
  labs(y = 'Probe Out of Sample Accuracy', 
       x = NULL,
       title = "Out of sample accuracy with logistic regression",
       subtitle = "Results on gemma 2b. Error bars indicate randomness from using different seeds",
       caption = "Loss calculated with BCEWithLogitsLoss in PyTorch",
       color = "Probe Location", shape = "Probe Location") + 
  geom_errorbar(aes(ymin = mean_test_accuracy - 1.96*se_test_accuracy, 
                    ymax = mean_test_accuracy + 1.96*se_test_accuracy),
                width = 0.1) +
  scale_y_continuous(labels = scales::percent) + 
  facet_wrap(~Dataset, scales = 'free') +
  theme(axis.text.x = element_blank(),  # Remove x-axis text
        axis.ticks.x = element_blank(),
        legend.position = 'bottom') +  # Remove x-axis ticks
  scale_color_manual(values = c(nicepurp, niceblue, nicegreen))
ggsave("reports/figures/logistic_accuracy_llama.png", width = 6, height = 4, scale = 1.6, dpi = 400)

ggplot(combined_plot, aes(x = graph_labels, 
                          y = mean_test_ROC,
                          color = graph_labels,
                          shape = graph_labels)) +  # Add color aesthetic
  geom_point() + 
  myTheme + 
  labs(y = 'Probe Out of Sample ROC AUC', 
       x = NULL,
       title = "Out of sample ROC AUC with logistic regression",
       subtitle = "Results on gemma 2b. Error bars indicate randomness from using different seeds",
       color = "Probe Location", shape = "Probe Location") + 
  geom_errorbar(aes(ymin = mean_test_ROC - 1.96*se_test_ROC, 
                    ymax = mean_test_ROC + 1.96*se_test_ROC),
                width = 0.1) +
  scale_y_continuous() + 
  facet_wrap(~Dataset, scales = 'free') +
  theme(axis.text.x = element_blank(),  # Remove x-axis text
        axis.ticks.x = element_blank(),
        legend.position = 'bottom') +  # Remove x-axis ticks
  scale_color_manual(values = c(nicepurp, niceblue, nicegreen))
ggsave("reports/figures/logistic_roc_auc_llama.png", width = 6, height = 4, scale = 1.2, dpi = 400)

pairwise_performance <- function(some_probes,n_sims = 100){
  colnames(some_probes) <- gsub(" ", "_", colnames(some_probes))
  
}

#####

#Truth probe

colnames(basketball) <- gsub(" ", "_", colnames(basketball))


headline_probes %>% select(Seed, Feature_Type, Test_Loss, Test_Accuracy) %>% 
  pivot_wider(id_cols = Seed, names_from = Feature_Type, 
              values_from = c( Test_Loss, Test_Accuracy)) %>%
  mutate(error_beats_input_loss = (Test_Loss_sae_diff - Test_Loss_sae_input) <0 ,
         error_beats_recon_loss = (Test_Loss_sae_diff - Test_Loss_sae_recons)<0,
         error_beats_input_accuracy = (Test_Accuracy_sae_diff - Test_Accuracy_sae_input) >0 ,
         error_beats_recon_accuracy = (Test_Accuracy_sae_diff - Test_Accuracy_sae_recons)>0,) %>%
  skim()

headline_probes %>% select(Seed, Feature_Type, Test_Loss, Test_ROC_AUC) %>% 
  pivot_wider(id_cols = Seed, names_from = Feature_Type, 
              values_from = c( Test_Loss, Test_ROC_AUC)) %>%
  mutate(error_beats_input_loss = (Test_Loss_sae_diff - Test_Loss_sae_input) <0 ,
         error_beats_recon_loss = (Test_Loss_sae_diff - Test_Loss_sae_recons)<0,
         error_beats_input_auc = (Test_ROC_AUC_sae_diff - Test_ROC_AUC_sae_input) >0 ,
         error_beats_recon_auc = (Test_ROC_AUC_sae_diff - Test_ROC_AUC_sae_recons)>0,) %>%
  skim()


truth_probes %<>% mutate(error_beats_input = )

#Data summaries

#load the 

df <- read_csv("results/meta_llama_Llama_3.1_8B_l19r_8x/probe_results_ath_basketball.csv")
df_sum <- probe_summarizer('Llama 3.1 Athelete Plays Basketball Probe',df)


truth_probes <- read_csv('results/meta_llama_Llama_3.1_8B_l19r_8x/probe_results_truth.csv')
truth_summary <- probe_summarizer('Probing for Truth in Cities Dataset', truth_probes)

headline_probes_old <- read_csv('data/processed/logistic_combined_gemma_2_2b_layer_19_width_16k_canonical_old/probe_results_headline_fp.csv')
headline_probes_old %>% group_by(Feature_Type) %>% skim()

headline_summary <- probe_summarizer('Probing for Headline (Front Page)', headline_probes)

manhattan_probes <- read_csv('results/meta_llama_Llama_3.1_8B_l19r_8x/probe_results_man_borough.csv')
manhattan_summary <- probe_summarizer('Probing for in Manhattan', manhattan_probes)

tw_happy <- read_csv("data/processed/logistic_combined_gemma_2_2b_layer_19_width_16k_canonical/probe_results_twt_happy.csv")
tw_happy %>% group_by(Feature_Type) %>% skim()

tw_happy_summary <- probe_summarizer('Probing for Happiness in Tweets', tw_happy)

basketball <- read_csv("results/meta_llama_Llama_3.1_8B_l19r_8x/probe_results_ath_basketball.csv")
basketball_summary <- probe_summarizer('Probing for Basketball Atheletes', basketball, save_fig = FALSE)

combined_plot <- truth_summary %>% mutate(Dataset = "Truth") %>% 
  add_row(headline_summary %>% mutate(Dataset = "Frontpage headlines")) %>%
  add_row(manhattan_summary %>% mutate(Dataset = "Location in Manhattan")) %>%
  add_row(tw_happy_summary %>% mutate(Dataset = "Happiness in Tweet")) %>%
  add_row(basketball_summary %>% mutate(Dataset = "Athelete plays basketball"))

ggplot(combined_plot, aes(x = graph_labels, 
                          y = mean_test_loss,
                          color = graph_labels,
                          shape = graph_labels)) +  # Add color aesthetic
  geom_point() + 
  myTheme + 
  labs(y = 'Probe Out of Sample Loss', 
       x = NULL,
       title = "Out of sample loss is lower on SAE Error across settings",
       subtitle = "Results on LLama 3.1 8B. Error bars indicate randomness from using different seeds",
       caption = "Loss calculated with BCEWithLogitsLoss in PyTorch",
       color = "Probe Location", shape = "Probe Location") + 
  geom_errorbar(aes(ymin = mean_test_loss - 1.96*se_test_loss, 
                    ymax = mean_test_loss + 1.96*se_test_loss),
                width = 0.1) +
  scale_y_continuous() + 
  facet_wrap(~Dataset, scales = 'free') +
  theme(axis.text.x = element_blank(),  # Remove x-axis text
        axis.ticks.x = element_blank(),
        legend.position = 'bottom') +  # Remove x-axis ticks
  scale_color_manual(values = c(nicepurp, niceblue, nicegreen))


ggplot(combined_plot, aes(x = graph_labels, 
                          y = mean_test_ROC,
                          color = graph_labels,
                          shape = graph_labels)) +  # Add color aesthetic
  geom_point() + 
  myTheme + 
  labs(y = 'Probe Out of Sample ROC AUC', 
       x = NULL,
       title = "Out of sample ROC AUC is a bit better when probing on SAE Error",
       subtitle = "Results on Llama 3.1 8B. Error bars indicate randomness from using different seeds",
       color = "Probe Location", shape = "Probe Location") + 
  geom_errorbar(aes(ymin = mean_test_ROC - 1.96*se_test_ROC, 
                    ymax = mean_test_ROC + 1.96*se_test_ROC),
                width = 0.1) +
  scale_y_continuous() + 
  facet_wrap(~Dataset, scales = 'free') +
  theme(axis.text.x = element_blank(),  # Remove x-axis text
        axis.ticks.x = element_blank(),
        legend.position = 'bottom') +  # Remove x-axis ticks
  scale_color_manual(values = c(nicepurp, niceblue, nicegreen))





truth_df <- read_csv('all_cities.csv')
View(slice_sample(truth_df, n= 10))

truth_probes <- read_csv('probe_results_truth.csv')
truth_summary <- probe_summarizer('Probing for Truth in Cities Dataset', truth_probes)

truth_probes_2nd <- read_csv('probe_results_truth_second_last.csv')
truth_2nd_summary <- probe_summarizer('Probing for Truth (2nd last token)', truth_probes_2nd)

headline_probes <- read_csv('probe_results_hl_frontp.csv')
headline_summary <- probe_summarizer('Probing for Headline (Front Page)', headline_probes)

manhattan_probes <- read_csv('probe_results_man_borough.csv')
manhattan_summary <- probe_summarizer('Probing for in Manhattan', manhattan_probes)

tw_happy <- read_csv('probe_results_tw_happiness.csv')
tw_happy_summary <- probe_summarizer('Probing for Happiness in Tweets', tw_happy)

basketball <- read_csv('probe_results_ath_sport.csv')
basketball_summary <- probe_summarizer('Probing for Basketball Atheletes', basketball)

twoshot <- read_csv('probe_results_twoshot.csv')
twoshot_summary <- probe_summarizer('Probing for Truth with Two shot prompt', twoshot)


steering_results_truth <- read_csv("~/OneDrive/Coding/AISC/SAE_Error_probes/steering_results_truth.csv")
feols(Steered_Logit_Diff ~ as.factor(Scaling_Factor) | Sample_Index, 
      data = steering_results_truth %>% filter(Feature_Type == "sae_input"), vcov = ~Sample_Index)

feols(Steered_Logit_Diff ~ as.factor(Scaling_Factor) | Sample_Index, 
      data = steering_results_truth %>% filter(Feature_Type == "sae_recons"), vcov = ~Sample_Index)

feols(Steered_Logit_Diff ~ as.factor(Scaling_Factor) | Sample_Index, 
      data = steering_results_truth %>% filter(Feature_Type == "sae_diff"), vcov = ~Sample_Index)

mean(abs(steering_results_truth$Baseline_Logit_Diff))

combined_plot <- truth_summary %>% mutate(Dataset = "Truth") %>% 
  add_row(truth_2nd_summary %>% mutate(Dataset = "Truth (second last token)")) %>%
  add_row(headline_summary %>% mutate(Dataset = "Frontpage headlines")) %>%
  add_row(manhattan_summary %>% mutate(Dataset = "Location in Manhattan")) %>%
  add_row(tw_happy_summary %>% mutate(Dataset = "Happiness in Tweet")) %>%
  add_row(basketball_summary %>% mutate(Dataset = "Athelete plays basketball"))
  
  
ggplot(combined_plot, aes(x = graph_labels, 
                          y = mean_test_loss,
                          color = graph_labels,
                          shape = graph_labels)) +  # Add color aesthetic
  geom_point() + 
  myTheme + 
  labs(y = 'Probe Out of Sample Loss', 
       x = NULL,
       title = "Out of sample loss is lower on SAE Error across settings",
       subtitle = "Error bars indicate randomness from using different seeds",
       caption = "Loss calculated with BCEWithLogitsLoss in PyTorch",
       color = "Probe Location", shape = "Probe Location") + 
  geom_errorbar(aes(ymin = mean_test_loss - 1.96*se_test_loss, 
                    ymax = mean_test_loss + 1.96*se_test_loss),
                width = 0.1) +
  scale_y_continuous() + 
  facet_wrap(~Dataset) +
  theme(axis.text.x = element_blank(),  # Remove x-axis text
        axis.ticks.x = element_blank(),
        legend.position = 'bottom') +  # Remove x-axis ticks
  scale_color_manual(values = c(nicepurp, niceblue, nicegreen))
  



ggplot(combined_plot, aes(x = graph_labels, 
                          y = mean_test_accuracy,
                          color = graph_labels,
                          shape = graph_labels)) +  # Add color aesthetic
  geom_point() + 
  myTheme + 
  labs(y = 'Probe Out of Sample Accuracy', 
       x = NULL,
       title = "Accuracy is higher on SAE Error in most settings",
       subtitle = "Error bars indicate randomness from using different seeds",
       color = "Probe Location", shape = "Probe Location") + 
  geom_errorbar(aes(ymin = mean_test_accuracy - 1.96*se_test_accuracy, 
                    ymax = mean_test_accuracy + 1.96*se_test_accuracy),
                width = 0.1) +
  scale_y_continuous(labels = scales::percent) + 
  facet_wrap(~Dataset) +
  theme(axis.text.x = element_blank(),  # Remove x-axis text
        axis.ticks.x = element_blank(),
        legend.position = 'bottom') +  # Remove x-axis ticks
  scale_color_manual(values = c(nicepurp, niceblue, nicegreen))

steering_summary <- steering_results_truth %>% group_by(Scaling_Factor, Feature_Type) %>%
  summarise(mean_logit_diff = mean(Steered_Logit_Diff),
            se_logit_diff = sd(Steered_Logit_Diff)/sqrt(n()))

steering_summary %<>% mutate(Probe_Location = case_when(Feature_Type == 'sae_diff' ~ 'SAE Error',
                                                       Feature_Type == 'sae_input' ~ 'Residual Stream',
                                                       TRUE ~ 'SAE Reconstruction'))

steering_summary %>% ungroup() %>% ggplot(aes(x = Scaling_Factor, y = mean_logit_diff)) +
  geom_point() + facet_wrap(~Probe_Location) +
  geom_errorbar(aes(ymin = mean_logit_diff - 1.96*se_logit_diff, 
                    ymax = mean_logit_diff + 1.96*se_logit_diff),
                width = 2) +
  labs(title = "Sterering using SAE error probes have less downstream effects on token logits",
       subtitle = "Mean absolute logit difference without steering is 0.598",
       y = "Logit difference between ' True' and ' False' tokens",
       x = "Scaling factor on the probe for steering",
       caption = "These standard errors are large partially because I'm steering on 420 different pieces of text") +
  myTheme



residual_probe_dot_products <- read_csv("residual_probe_dot_products.csv")

residual_probe_dot_products %>% select(sae_input_dot_product, 
                                       sae_recons_dot_product, sae_diff_dot_product) %>%
  pivot_longer(cols = c(sae_input_dot_product, 
                        sae_recons_dot_product, sae_diff_dot_product),
               names_to = 'probeloc') %>% 
  mutate(Probe_Location = case_when(
    probeloc == 'sae_input_dot_product' ~ 'Residual',
    probeloc == 'sae_recons_dot_product' ~ 'SAE Reconstruction',
    TRUE ~ 'SAE Error'
  )) %>%
  ggplot() + geom_histogram(aes(value), binwidth = 2, center = 0, fill = niceblue) + 
  myTheme + 
  facet_wrap(~Probe_Location) +
  labs(title = "Dot products along the trained probe direction",
       caption = "Probe norm normalized to one",
       subtitle = "Dot products calculated for 20 probes in each category on 128 prompts",
       x = "Dot products", y = "Count")





