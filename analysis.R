# analysis.R  —  Music Genre Classification Results
# Run from project root with RStudio open to this project.
# All figures saved to figures/

# ── 0. Packages ───────────────────────────────────────────────────────────────

pkgs <- c("tidyverse", "jsonlite", "scales", "viridis", "patchwork", "reshape2")
new  <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(new)) install.packages(new, repos = "https://cloud.r-project.org")

library(tidyverse)
library(jsonlite)
library(scales)
library(viridis)
library(patchwork)
library(reshape2)

theme_set(theme_minimal(base_size = 13))
dir.create("figures", showWarnings = FALSE)

GENRE_NAMES <- c("Blues","Classical","Country","Easy Listening","Electronic",
                 "Experimental","Folk","Hip-Hop","Instrumental","International",
                 "Jazz","Old-Time / Historic","Pop","Rock","Soul-RnB","Spoken")

# ── 1. Dataset Comparison ─────────────────────────────────────────────────────

datasets <- tribble(
  ~Dataset,      ~Tracks,  ~Genres, ~Label_Scheme,
  "GTZAN",          1000,      10,  "Balanced (100/genre)",
  "FMA Small",      8000,       8,  "Balanced (1000/genre)",
  "FMA Medium",    25000,      16,  "Imbalanced (18-6099)",
  "FMA Large",    106574,     161,  "Imbalanced"
)

p1a <- ggplot(datasets, aes(x = reorder(Dataset, Tracks), y = Tracks, fill = Dataset)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = comma(Tracks)), hjust = -0.1, size = 4) +
  coord_flip() +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.18))) +
  scale_fill_viridis_d(option = "C", begin = 0.2, end = 0.85) +
  labs(title = "Number of Tracks", x = NULL, y = "Tracks") +
  theme(panel.grid.major.y = element_blank())

p1b <- ggplot(datasets, aes(x = reorder(Dataset, Genres), y = Genres, fill = Dataset)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = Genres), hjust = -0.1, size = 4) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  scale_fill_viridis_d(option = "C", begin = 0.2, end = 0.85) +
  labs(title = "Number of Genres", x = NULL, y = "Genres") +
  theme(panel.grid.major.y = element_blank())

p1 <- p1a + p1b +
  plot_annotation(
    title    = "Music Genre Classification Datasets",
    subtitle = "FMA Medium chosen: real-world scale with genre imbalance challenge",
    theme    = theme(plot.title = element_text(size = 16, face = "bold"))
  )
ggsave("figures/01_dataset_comparison.png", p1, width = 13, height = 5, dpi = 150)
message("Saved 01_dataset_comparison.png")

# ── 2. Preprocessing Pipeline ─────────────────────────────────────────────────

prep <- fromJSON("data/preprocessing_log.json")

class_df <- enframe(unlist(prep$class_distribution), name = "Genre", value = "Count") %>%
  mutate(Count = as.integer(Count))

p2a <- ggplot(class_df, aes(x = reorder(Genre, Count), y = Count, fill = Count)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = comma(Count)), hjust = -0.1, size = 3.4) +
  coord_flip() +
  scale_fill_viridis_c(option = "C", begin = 0.15, end = 0.9) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Class Distribution - FMA Medium",
    subtitle = sprintf(
      "%s tracks processed, %d skipped | Rock + Electronic = %.0f%% of data",
      comma(prep$tracks_processed),
      prep$tracks_skipped,
      100 * (class_df$Count[class_df$Genre == "Rock"] +
               class_df$Count[class_df$Genre == "Electronic"]) / sum(class_df$Count)
    ),
    x = NULL, y = "Track Count"
  )

stats_df <- tibble(
  Metric = c("Total Time (min)", "Workers Used", "Peak RAM (GB)", "Avg Time / Track (s)"),
  Value  = c(
    prep$total_time_min,
    prep$workers,
    round(prep$peak_memory_mb / 1024, 1),
    prep$per_track_time_sec$mean
  )
)

p2b <- ggplot(stats_df, aes(x = reorder(Metric, Value), y = Value, fill = Metric)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = round(Value, 2)), hjust = -0.15, size = 4) +
  coord_flip() +
  scale_fill_viridis_d(option = "D", begin = 0.2, end = 0.8) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(title = "Pipeline Performance", x = NULL, y = "Value") +
  theme(panel.grid.major.y = element_blank())

p2 <- p2a + p2b + plot_layout(widths = c(2, 1.2)) +
  plot_annotation(
    title = "Step 1: Preprocessing Pipeline",
    theme = theme(plot.title = element_text(size = 16, face = "bold"))
  )
ggsave("figures/02_preprocessing.png", p2, width = 15, height = 6, dpi = 150)
message("Saved 02_preprocessing.png")

# ── 3. Naive Baseline ─────────────────────────────────────────────────────────

naive <- read_csv("results/baseline_overfit_20260506_162852/epoch_metrics.csv",
                  show_col_types = FALSE)

p3a <- naive %>%
  select(epoch, train_acc, val_acc) %>%
  pivot_longer(-epoch, names_to = "Split", values_to = "Accuracy") %>%
  mutate(Split = recode(Split, train_acc = "Train", val_acc = "Validation")) %>%
  ggplot(aes(x = epoch, y = Accuracy, color = Split)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 1.5) +
  scale_color_manual(values = c(Train = "#1565C0", Validation = "#C62828")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "Train vs Validation Accuracy",
       subtitle = "Train climbs steadily; val flatlines ~epoch 15 -> overfitting",
       x = "Epoch", y = "Accuracy", color = NULL) +
  theme(legend.position = "bottom")

p3b <- naive %>%
  ggplot(aes(x = epoch, y = train_loss)) +
  geom_line(color = "#E65100", linewidth = 1.1) +
  geom_point(size = 1.5, color = "#E65100") +
  labs(title = "Training Loss", x = "Epoch", y = "Cross-Entropy Loss")

p3c <- naive %>%
  filter(epoch_time_sec < 200) %>%
  ggplot(aes(x = epoch, y = epoch_time_sec)) +
  geom_col(fill = "#6A1B9A", alpha = 0.75) +
  labs(title = "Epoch Wall-Clock Time (s)", x = "Epoch", y = "Seconds")

p3 <- (p3a | p3b) / p3c +
  plot_annotation(
    title = "Step 2: Naive Baseline (lr=1e-3, bs=128, rnn=128, 50 epochs)",
    theme = theme(plot.title = element_text(size = 16, face = "bold"))
  )
ggsave("figures/03_naive_baseline.png", p3, width = 14, height = 9, dpi = 150)
message("Saved 03_naive_baseline.png")

# ── 4. Hyperparameter Tuning ──────────────────────────────────────────────────

load_phase_runs <- function(base, subdirs, param_name) {
  map_dfr(names(subdirs), function(subdir) {
    read_csv(file.path(base, subdir, "epoch_metrics.csv"), show_col_types = FALSE) %>%
      mutate(param = subdirs[[subdir]])
  }) %>% rename(!!param_name := param)
}

# Phase 1: Learning Rate
lr_map <- c("lr_1e-04" = "0.0001", "lr_3e-04" = "0.0003", "lr_1e-03" = "0.001",
            "lr_3e-03" = "0.003",  "lr_1e-02" = "0.01")
phase1 <- load_phase_runs("results/tuning_20260506_180753/phase1", lr_map, "Learning Rate") %>%
  mutate(`Learning Rate` = factor(`Learning Rate`, levels = c("0.0001","0.0003","0.001","0.003","0.01")))

p4a <- ggplot(phase1, aes(x = epoch, y = val_acc, color = `Learning Rate`)) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.2) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_color_viridis_d(option = "C", begin = 0.1, end = 0.9) +
  labs(title = "Phase 1: Learning Rate", x = "Epoch", y = "Val Accuracy", color = "LR") +
  theme(legend.position = "bottom")

# Phase 2: Batch Size
bs_map <- c("bs_32" = "32", "bs_64" = "64", "bs_128" = "128")
phase2 <- load_phase_runs("results/tuning_20260506_180753/phase2", bs_map, "Batch Size") %>%
  mutate(`Batch Size` = factor(`Batch Size`, levels = c("32","64","128")))

p4b <- ggplot(phase2, aes(x = epoch, y = val_acc, color = `Batch Size`)) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.2) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_color_manual(values = c("32" = "#1565C0", "64" = "#2E7D32", "128" = "#E65100")) +
  labs(title = "Phase 2: Batch Size", x = "Epoch", y = "Val Accuracy", color = "Batch") +
  theme(legend.position = "bottom")

# Phase 3: RNN Hidden Units (filter PC-sleep anomaly epoch in rnn_256)
rnn_map <- c("rnn_64" = "64", "rnn_128" = "128", "rnn_256" = "256", "rnn_512" = "512")
phase3 <- load_phase_runs("results/tuning_20260506_180753/phase3", rnn_map, "RNN Hidden") %>%
  mutate(`RNN Hidden` = factor(`RNN Hidden`, levels = c("64","128","256","512"))) %>%
  filter(epoch_time_sec < 1000)

p4c <- ggplot(phase3, aes(x = epoch, y = val_acc, color = `RNN Hidden`)) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.2) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_color_manual(values = c("64"="#7B1FA2","128"="#1565C0","256"="#2E7D32","512"="#C62828")) +
  labs(title = "Phase 3: RNN Hidden Units", x = "Epoch", y = "Val Accuracy", color = "Hidden") +
  theme(legend.position = "bottom")

p4 <- p4a | p4b | p4c +
  plot_annotation(
    title = "Step 3: Hyperparameter Tuning - Validation Accuracy per Phase (25 epochs each)",
    theme = theme(plot.title = element_text(size = 16, face = "bold"))
  )
ggsave("figures/04_hyperparameter_phases.png", p4, width = 17, height = 6, dpi = 150)
message("Saved 04_hyperparameter_phases.png")

# Average val accuracy per parameter value across all epochs (faceted by phase)
avg_p1 <- phase1 %>%
  group_by(value = `Learning Rate`) %>%
  summarise(mean_val = mean(val_acc), sd_val = sd(val_acc), .groups = "drop") %>%
  mutate(Phase = "Phase 1: Learning Rate")

avg_p2 <- phase2 %>%
  group_by(value = `Batch Size`) %>%
  summarise(mean_val = mean(val_acc), sd_val = sd(val_acc), .groups = "drop") %>%
  mutate(Phase = "Phase 2: Batch Size")

avg_p3 <- phase3 %>%
  group_by(value = `RNN Hidden`) %>%
  summarise(mean_val = mean(val_acc), sd_val = sd(val_acc), .groups = "drop") %>%
  mutate(Phase = "Phase 3: RNN Hidden Units")

avg_all <- bind_rows(avg_p1, avg_p2, avg_p3) %>%
  mutate(Phase = factor(Phase, levels = c("Phase 1: Learning Rate",
                                           "Phase 2: Batch Size",
                                           "Phase 3: RNN Hidden Units")))

p4d <- ggplot(avg_all, aes(x = value, y = mean_val, fill = Phase)) +
  geom_col(show.legend = FALSE, width = 0.6) +
  geom_errorbar(aes(ymin = mean_val - sd_val, ymax = mean_val + sd_val),
                width = 0.25, color = "grey30", linewidth = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", mean_val * 100)),
            vjust = -1.6, size = 3.5, fontface = "bold") +
  facet_wrap(~Phase, scales = "free_x") +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0.02, 0.12))) +
  scale_fill_manual(values = c("Phase 1: Learning Rate"     = "#1565C0",
                                "Phase 2: Batch Size"        = "#2E7D32",
                                "Phase 3: RNN Hidden Units"  = "#C62828")) +
  labs(title    = "Average Validation Accuracy per Parameter Value (across all 25 epochs)",
       subtitle = "Error bars show +/- 1 SD. Higher and tighter = better and more stable.",
       x = "Parameter Value", y = "Mean Validation Accuracy") +
  theme(strip.text = element_text(face = "bold", size = 11),
        panel.grid.major.x = element_blank())
ggsave("figures/04b_tuning_avg_performance.png", p4d, width = 13, height = 6, dpi = 150)
message("Saved 04b_tuning_avg_performance.png")

# ── 5. Optimized Final Runs ───────────────────────────────────────────────────

no_wt <- read_csv("results/optimized_no_weights_20260506_214806/epoch_metrics.csv",
                  show_col_types = FALSE) %>% mutate(Run = "No Class Weights")
wt    <- read_csv("results/optimized_weighted_20260507_172654/epoch_metrics.csv",
                  show_col_types = FALSE) %>% mutate(Run = "Inverse-Freq Weighted")
final <- bind_rows(no_wt, wt)

p5a <- final %>%
  select(epoch, train_acc, val_acc, Run) %>%
  pivot_longer(c(train_acc, val_acc), names_to = "Split", values_to = "Accuracy") %>%
  mutate(Split = recode(Split, train_acc = "Train", val_acc = "Validation")) %>%
  ggplot(aes(x = epoch, y = Accuracy, color = Split)) +
  geom_line(linewidth = 1) + geom_point(size = 0.9) +
  facet_wrap(~Run) +
  scale_color_manual(values = c(Train = "#1565C0", Validation = "#C62828")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "Train vs Validation Accuracy (50 epochs)",
       x = "Epoch", y = "Accuracy", color = NULL) +
  theme(legend.position = "bottom", strip.text = element_text(face = "bold"))

p5b <- final %>%
  ggplot(aes(x = epoch, y = train_loss, color = Run)) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c("No Class Weights" = "#1565C0",
                                 "Inverse-Freq Weighted" = "#C62828")) +
  labs(title = "Training Loss Comparison",
       x = "Epoch", y = "Cross-Entropy Loss", color = NULL) +
  theme(legend.position = "bottom")

p5 <- p5a / p5b +
  plot_annotation(
    title = "Step 4: Optimized Runs (lr=3e-4, bs=64, rnn=512)",
    theme = theme(plot.title = element_text(size = 16, face = "bold"))
  )
ggsave("figures/05_optimized_runs.png", p5, width = 14, height = 10, dpi = 150)
message("Saved 05_optimized_runs.png")

# Summary metric comparison
read_metrics <- function(run_dir) {
  sr <- fromJSON(file.path(run_dir, "run_summary.json"))
  vm <- fromJSON(file.path(run_dir, "val_metrics.json"))
  list(val_acc = sr$best_val_acc, macro_f1 = vm$macro_f1, balanced_acc = vm$balanced_accuracy)
}

m_naive <- read_metrics("results/baseline_overfit_20260506_162852")
m_nowt  <- read_metrics("results/optimized_no_weights_20260506_214806")
m_wt    <- read_metrics("results/optimized_weighted_20260507_172654")

metric_cmp <- tribble(
  ~Run,                    ~`Val Accuracy`, ~`Macro F1`, ~`Balanced Accuracy`,
  "Naive Baseline",        m_naive$val_acc, m_naive$macro_f1, m_naive$balanced_acc,
  "Optimized (No Wts)",    m_nowt$val_acc,  m_nowt$macro_f1,  m_nowt$balanced_acc,
  "Optimized (Weighted)",  m_wt$val_acc,    m_wt$macro_f1,    m_wt$balanced_acc,
) %>%
  mutate(Run = factor(Run, levels = c("Naive Baseline","Optimized (No Wts)","Optimized (Weighted)")))

p5c <- metric_cmp %>%
  pivot_longer(-Run, names_to = "Metric", values_to = "Score") %>%
  ggplot(aes(x = Run, y = Score, fill = Metric)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = sprintf("%.3f", Score)),
            position = position_dodge(0.9), vjust = -0.4, size = 3.3) +
  scale_fill_manual(values = c("Val Accuracy"      = "#1565C0",
                                "Macro F1"          = "#2E7D32",
                                "Balanced Accuracy" = "#E65100")) +
  scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
  labs(title    = "All Runs: Final Metric Comparison",
       subtitle = "Weighted loss raises Macro F1 and Balanced Accuracy at slight cost to raw Val Acc",
       x = NULL, y = "Score", fill = "Metric") +
  theme(legend.position = "bottom", axis.text.x = element_text(angle = 10, hjust = 1))
ggsave("figures/05b_metric_comparison.png", p5c, width = 10, height = 6, dpi = 150)
message("Saved 05b_metric_comparison.png")

# ── 6. Confusion Matrices ─────────────────────────────────────────────────────

read_cm_normalized <- function(path) {
  raw     <- read_csv(path, show_col_types = FALSE)
  mat     <- as.matrix(raw[, -1])
  totals  <- rowSums(mat)
  mat_pct <- sweep(mat, 1, ifelse(totals == 0, 1, totals), FUN = "/")
  rownames(mat_pct) <- GENRE_NAMES
  colnames(mat_pct) <- GENRE_NAMES
  melt(mat_pct, varnames = c("True", "Predicted"), value.name = "Recall")
}

cm_nowt <- read_cm_normalized("results/optimized_no_weights_20260506_214806/confusion_matrix.csv") %>%
  mutate(Run = "No Class Weights")
cm_wt   <- read_cm_normalized("results/optimized_weighted_20260507_172654/confusion_matrix.csv") %>%
  mutate(Run = "Inverse-Freq Weighted")
cm_all  <- bind_rows(cm_nowt, cm_wt)

p6 <- ggplot(cm_all, aes(x = Predicted, y = fct_rev(True), fill = Recall)) +
  geom_tile(color = "white", linewidth = 0.25) +
  facet_wrap(~Run) +
  scale_fill_viridis_c(option = "B", labels = percent_format(accuracy = 1),
                        limits = c(0, 1), name = "Recall") +
  labs(title    = "Confusion Matrices - Row-Normalized (Recall per Genre)",
       subtitle = "Diagonal = correctly classified; brighter = higher recall",
       x = "Predicted", y = "True") +
  theme(
    axis.text.x     = element_text(angle = 45, hjust = 1, size = 7),
    axis.text.y     = element_text(size = 7),
    strip.text      = element_text(face = "bold", size = 12),
    legend.position = "right"
  )
ggsave("figures/06_confusion_matrices.png", p6, width = 18, height = 8, dpi = 150)
message("Saved 06_confusion_matrices.png")

# ── 7. Per-class F1 comparison ────────────────────────────────────────────────

read_per_class_f1 <- function(run_dir, run_label) {
  vm       <- fromJSON(file.path(run_dir, "val_metrics.json"))
  pc       <- vm$per_class
  idx_keys <- names(pc)[!names(pc) %in% c("accuracy", "macro avg", "weighted avg")]
  tibble(
    class_idx = as.integer(idx_keys),
    f1        = map_dbl(idx_keys, ~ pc[[.x]]$`f1-score`),
    support   = map_dbl(idx_keys, ~ pc[[.x]]$support)
  ) %>%
    mutate(Genre = GENRE_NAMES[class_idx + 1], Run = run_label)
}

pc_nowt <- read_per_class_f1("results/optimized_no_weights_20260506_214806", "No Class Weights")
pc_wt   <- read_per_class_f1("results/optimized_weighted_20260507_172654",   "Inverse-Freq Weighted")
pc_all  <- bind_rows(pc_nowt, pc_wt)

p7 <- ggplot(pc_all, aes(x = reorder(Genre, f1), y = f1, fill = Run)) +
  geom_col(position = "dodge") +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey40") +
  coord_flip() +
  scale_fill_manual(values = c("No Class Weights"      = "#1565C0",
                                "Inverse-Freq Weighted" = "#C62828")) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0, 1), expand = expansion(mult = c(0, 0.05))) +
  labs(title    = "Per-Genre F1 Score: Weighted vs Unweighted",
       subtitle = "Dashed line = 0.50. Weighted loss helps minority genres most.",
       x = NULL, y = "F1 Score", fill = NULL) +
  theme(legend.position = "bottom", panel.grid.major.y = element_blank())
ggsave("figures/07_per_class_f1.png", p7, width = 11, height = 9, dpi = 150)
message("Saved 07_per_class_f1.png")

# ── 8. Class Weighting: Who Did It Help? ─────────────────────────────────────

get_diagonal_recall <- function(path) {
  raw     <- read_csv(path, show_col_types = FALSE)
  mat     <- as.matrix(raw[, -1])
  totals  <- rowSums(mat)
  mat_pct <- sweep(mat, 1, ifelse(totals == 0, 1, totals), FUN = "/")
  diag(mat_pct)
}

recall_nowt <- get_diagonal_recall("results/optimized_no_weights_20260506_214806/confusion_matrix.csv")
recall_wt   <- get_diagonal_recall("results/optimized_weighted_20260507_172654/confusion_matrix.csv")

delta_df <- tibble(
  Genre       = GENRE_NAMES,
  recall_nowt = recall_nowt,
  recall_wt   = recall_wt,
  delta       = recall_wt - recall_nowt
) %>%
  left_join(class_df, by = "Genre") %>%
  mutate(
    Outcome = case_when(
      delta >  0.02 ~ "Improved",
      delta < -0.02 ~ "Worsened",
      TRUE          ~ "Unchanged"
    ),
    Outcome = factor(Outcome, levels = c("Improved","Unchanged","Worsened"))
  )

# Nudge labels so they don't overlap points
delta_df <- delta_df %>%
  mutate(
    nudge_x = 0,
    nudge_y = case_when(
      Genre == "Blues"      ~  0.04,
      Genre == "Soul-RnB"   ~ -0.04,
      Genre == "Rock"       ~  0.04,
      Genre == "Electronic" ~ -0.04,
      TRUE                  ~  0.03
    )
  )

p8 <- ggplot(delta_df, aes(x = Count, y = delta, color = Outcome, label = Genre)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.8) +
  annotate("rect", xmin = 0, xmax = 120, ymin = -Inf, ymax = Inf,
           fill = "#FFCDD2", alpha = 0.25) +
  annotate("text", x = 60, y = 0.38, label = "Too rare\nto learn",
           color = "#C62828", size = 3.5, fontface = "italic", hjust = 0.5) +
  geom_smooth(method = "loess", se = TRUE, color = "grey60",
              fill = "grey85", linewidth = 0.8, show.legend = FALSE) +
  geom_point(size = 4.5, alpha = 0.9) +
  geom_text(aes(y = delta + nudge_y), size = 3.2, fontface = "bold",
            show.legend = FALSE) +
  scale_x_log10(labels = comma, breaks = c(20, 50, 100, 200, 500, 1000, 5000)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_color_manual(values = c("Improved"  = "#2E7D32",
                                 "Unchanged" = "#F57C00",
                                 "Worsened"  = "#C62828")) +
  labs(
    title    = "Did Inverse-Frequency Weighting Help Each Genre? (Metric: Recall)",
    subtitle = paste0("Recall = % of true songs correctly identified (confusion matrix diagonal). ",
                      "NOT the same as F1 or overall accuracy.\n",
                      "Genres in pink zone are too rare to learn regardless of weighting — ",
                      "no training examples = no signal."),
    x     = "Training Samples (log scale)",
    y     = "Recall Change (Weighted minus Unweighted)",
    color = NULL
  ) +
  theme(
    legend.position   = "bottom",
    plot.title        = element_text(size = 15, face = "bold"),
    plot.subtitle     = element_text(size = 10, color = "grey30"),
    panel.grid.minor  = element_blank()
  )

ggsave("figures/08_weighting_effect_by_class.png", p8, width = 11, height = 7, dpi = 150)
message("Saved 08_weighting_effect_by_class.png")

# ── 9. Recall Change by Sample Count Bucket ───────────────────────────────────

bucket_df <- delta_df %>%
  mutate(
    Bucket = cut(Count,
                 breaks = c(0, 100, 500, 1000, Inf),
                 labels = c("0–100\n(Very Rare)", "100–500\n(Rare)",
                            "500–1000\n(Moderate)", "1000+\n(Common)"),
                 include.lowest = TRUE)
  ) %>%
  group_by(Bucket) %>%
  summarise(
    mean_delta  = mean(delta),
    se          = sd(delta) / sqrt(n()),
    n_genres    = n(),
    genres      = paste(Genre, collapse = "\n"),
    .groups     = "drop"
  ) %>%
  mutate(Direction = ifelse(mean_delta >= 0, "Improved", "Worsened"))

p9 <- ggplot(bucket_df, aes(x = Bucket, y = mean_delta, fill = Direction)) +
  geom_col(width = 0.6) +
  geom_errorbar(aes(ymin = mean_delta - se, ymax = mean_delta + se),
                width = 0.2, color = "grey30", linewidth = 0.8) +
  geom_hline(yintercept = 0, linewidth = 0.8, color = "grey20") +
  geom_text(aes(
    label = sprintf("%+.1f%%", mean_delta * 100),
    vjust = ifelse(mean_delta >= 0, -0.6, 1.4)
  ), fontface = "bold", size = 5) +
  geom_text(aes(
    label = paste0("n=", n_genres, " genres"),
    y = 0,
    vjust = ifelse(mean_delta >= 0, 1.6, -0.8)
  ), color = "grey40", size = 3.3) +
  scale_fill_manual(values = c("Improved" = "#2E7D32", "Worsened" = "#C62828")) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0.15, 0.15))) +
  labs(
    title    = "Effect of Inverse-Frequency Weighting by Training Sample Count",
    subtitle = "Average recall change per genre, grouped by how many training samples that genre had.\nError bars = standard error across genres in each bucket.",
    x        = "Training Samples per Genre",
    y        = "Avg Recall Change (Weighted - Unweighted)",
    fill     = NULL
  ) +
  theme(
    legend.position  = "bottom",
    plot.title       = element_text(size = 14, face = "bold"),
    plot.subtitle    = element_text(size = 9.5, color = "grey30"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank()
  )

ggsave("figures/09_recall_change_by_bucket.png", p9, width = 10, height = 6, dpi = 150)
message("Saved 09_recall_change_by_bucket.png")

# ── 10. Precision vs Recall Tradeoff (Spoken case study) ─────────────────────

# Build per-class metrics table for both runs
read_all_class_metrics <- function(run_dir, run_label) {
  vm       <- fromJSON(file.path(run_dir, "val_metrics.json"))
  pc       <- vm$per_class
  idx_keys <- names(pc)[!names(pc) %in% c("accuracy", "macro avg", "weighted avg")]
  tibble(
    Genre     = GENRE_NAMES[as.integer(idx_keys) + 1],
    Precision = map_dbl(idx_keys, ~ pc[[.x]]$precision),
    Recall    = map_dbl(idx_keys, ~ pc[[.x]]$recall),
    F1        = map_dbl(idx_keys, ~ pc[[.x]]$`f1-score`),
    Run       = run_label
  )
}

all_class_metrics <- bind_rows(
  read_all_class_metrics("results/optimized_no_weights_20260506_214806", "No Class Weights"),
  read_all_class_metrics("results/optimized_weighted_20260507_172654",   "Inverse-Freq Weighted")
)

# Left panel: Spoken case study — all 3 metrics
spoken_df <- all_class_metrics %>%
  filter(Genre == "Spoken") %>%
  pivot_longer(c(Precision, Recall, F1), names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = factor(Metric, levels = c("Precision", "Recall", "F1")))

p10a <- ggplot(spoken_df, aes(x = Metric, y = Value, fill = Run)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", Value)),
            position = position_dodge(0.6), vjust = -0.5, size = 4, fontface = "bold") +
  annotate("segment", x = 0.85, xend = 1.15, y = 0.464, yend = 0.464,
           color = "#1565C0", linewidth = 0.6, linetype = "dotted") +
  annotate("segment", x = 0.85, xend = 1.15, y = 0.632, yend = 0.632,
           color = "#C62828", linewidth = 0.6, linetype = "dotted") +
  scale_fill_manual(values = c("No Class Weights" = "#1565C0",
                                "Inverse-Freq Weighted" = "#C62828")) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0, 0.85), expand = expansion(mult = c(0, 0.08))) +
  labs(title = "Spoken Word: Precision Up, Recall Down, F1 Up",
       x = NULL, y = "Score", fill = NULL) +
  theme(legend.position = "bottom", panel.grid.major.x = element_blank())

# Right panel: annotation slide explaining precision vs recall vs F1
annotation_data <- tribble(
  ~x, ~y, ~label, ~color,
  0.5, 0.85, "PRECISION", "grey20",
  0.5, 0.78, "\"When the model says Spoken,\nhow often is it right?\"", "grey40",
  0.5, 0.67, "No Weights: 46.4%  ->  Weighted: 63.2%  (+16.8%)", "#1565C0",
  0.5, 0.55, "RECALL", "grey20",
  0.5, 0.48, "\"Of all actual Spoken songs,\nhow many did the model find?\"", "grey40",
  0.5, 0.37, "No Weights: 59.1%  ->  Weighted: 54.5%  (-4.6%)", "#C62828",
  0.5, 0.25, "F1  (harmonic mean of both)", "grey20",
  0.5, 0.14, "No Weights: 52.0%  ->  Weighted: 58.5%  (+6.5%)", "#2E7D32",
)

p10b <- ggplot() +
  annotate("rect", xmin=0, xmax=1, ymin=0.59, ymax=0.98, fill="#E3F2FD", alpha=0.5, color="#90CAF9") +
  annotate("rect", xmin=0, xmax=1, ymin=0.29, ymax=0.58, fill="#FFEBEE", alpha=0.5, color="#EF9A9A") +
  annotate("rect", xmin=0, xmax=1, ymin=0.05, ymax=0.28, fill="#E8F5E9", alpha=0.5, color="#A5D6A7") +
  geom_text(data = annotation_data,
            aes(x=x, y=y, label=label, color=color),
            size = c(4.2, 3.5, 3.5, 4.2, 3.5, 3.5, 4.2, 3.5),
            fontface = c("bold","plain","plain","bold","plain","plain","bold","plain"),
            lineheight = 0.95) +
  scale_color_identity() +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "Why Can These Move in Different Directions?",
       subtitle = "Weighting made the model more selective (higher precision)\nat the cost of missing more true Spoken songs (lower recall).\nF1 captures the net result.") +
  theme_void() +
  theme(plot.title    = element_text(size = 12, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 9, color = "grey30", hjust = 0.5,
                                     margin = margin(t=4, b=8)))

p10 <- p10a + p10b +
  plot_annotation(
    title = "Case Study: Spoken Word — Precision vs Recall vs F1",
    theme = theme(plot.title = element_text(size = 15, face = "bold"))
  )
ggsave("figures/10_spoken_precision_recall_tradeoff.png", p10, width = 13, height = 7, dpi = 150)
message("Saved 10_spoken_precision_recall_tradeoff.png")

# ── Done ──────────────────────────────────────────────────────────────────────

message("\nAll figures written to figures/:")
walk(list.files("figures", full.names = FALSE), ~ message("  ", .x))
