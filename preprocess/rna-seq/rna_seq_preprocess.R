if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

R.version.string
#BiocManager::install(version = "3.18")
setwd("/Users/wanzi/Desktop/medicine_project")
getwd()

#BiocManager::install("biomaRt")
#BiocManager::install("readr")

library(biomaRt)
library(readr)

#filter out genes that target Y chromosome 
rna_seq <- read_csv("TCGA-COAD.htseq_fpkm_1.csv", show_col_types = FALSE)
rna_gene_list <- rna_seq$Ensembl_ID
#remove version number
rna_gene_list_no_version <- gsub("\\..*", "", rna_gene_list)
mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
gene_info <- getBM(attributes = c('ensembl_gene_id', 'chromosome_name'),
                   filters = 'ensembl_gene_id',
                   values = rna_gene_list_no_version,
                   mart = mart)
filtered_genes <- gene_info[!gene_info$chromosome_name %in% 'Y', ]
filtered_gene_list <- filtered_genes$ensembl_gene_id

#save to a csv file
write.csv(filtered_gene_list, "rna_seq_filtered_y_chromosome.csv", row.names = FALSE, quote = FALSE)
