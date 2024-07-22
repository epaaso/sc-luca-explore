args <- commandArgs(TRUE)

# Specify a CRAN mirror
cranMirror <- "https://cran.itam.mx/"
options(warn=-1)

# Local and global paths, where the local input is passed as an argument 
sampleName <- args[1] # Pass the CellRanger file as an argument 
globalPath <- "/root/datos/maestria/netopaas/Zuani2024/" # REPLACE WITH YOUR PATH TO THE TOP DIRECTORY CONTAINING ALL CELLRANGER OUTPUTS
samplePath <- paste(globalPath, sampleName, sep="")

fname <- paste(globalPath, sampleName, sep="")

pltPath <- paste(globalPath, '/drop_plots/', sep="") # PATH FOR PLOTS
if (!dir.exists(pltPath)) {
    print(paste("Creating folder", pltPath))
    dir.create(file.path(pltPath)) 
}

# We will store the results in each CellRanger directory, in a new subdirector called outputEmptyDrops
fname2 <- paste(globalPath, "/outputEmptyDrops", sep="")
fname2 <- paste(fname2, sampleName, sep='/')
print(fname2)
# fname2 <- paste(globalPath, subdirPath, sep="")
if (dir.exists(fname2)) {
    print("Folder /outputEmptyDrops already exist!")
} else {
    print(paste("Creating folder ", fname2))
    dir.create(file.path(fname2)) 
}

library(knitr)

opts_chunk$set(error=FALSE, message=FALSE, warning=FALSE)
opts_chunk$set(dpi=300, dev="png", dev.args=list(pointsize=15))
options(bitmapType="cairo")

# Install DropletUtils via the BiocManager
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager", repos=cranMirror)

# if (!requireNamespace("DropletUtils", quietly = TRUE))
#   BiocManager::install("DropletUtils")
library(DropletUtils)

suppressPackageStartupMessages(library(DropletUtils))

# Read from the 10X counts matrix
sce   <- read10xCounts(fname, col.names=TRUE, type='prefix')
set.seed(100)

# Cell barcode ranking
bcrank = NULL
# Alternative ranking if a lower bound is specified
if (grepl("LowerBound", sampleName, fixed=TRUE)) {
    print("Analysing lower bound inputs")
    bcrank = barcodeRanks(counts(sce), lower=500)
} else {
    bcrank = barcodeRanks(counts(sce))
}

# Running EmptyDrops
e.out  <- emptyDrops(counts(sce))
# Keep cell rows of the matix with low false discovery rate
is.cell = (e.out$FDR <= 0.01)

# Only showing unique points for plotting speed.
# uniq <- duplicated(bcrank$rank)
# plt1 <- paste(pltPath, "Rank_vs_UMI_", sampleName, ".png", sep="")
# png(file=plt1)
# par(mar=c(5,4,2,1), bty="n")
# plot(bcrank$rank[uniq], bcrank$total[uniq], log="xy", xlab="Rank", ylab="Total UMI count", cex=0.5, cex.lab=1.2)
# abline(h=metadata(bcrank)$inflection, col="darkgreen", lty=2)
# abline(h=metadata(bcrank)$knee, col="dodgerblue", lty=2)
# legend("left", legend=c("Inflection", "Knee"), bty="n", col=c("darkgreen", "dodgerblue"), lty=2, cex=1.2)
# dev.off()

# # Plotting after cell preselection
# plt2 <- paste(pltPath, "UMI_vs_LogProb_", sampleName, ".png", sep="")
# png(file=plt2)
# par(mar=c(5,4,1,1), mfrow=c(1,2), bty="n")
# plot(e.out$Total, -e.out$LogProb, col=ifelse(is.cell, "red", "black"), xlab="Total UMI count", ylab="-Log Probability", cex=0.5)
# abline(v = metadata(bcrank)$inflection, col="darkgreen")
# abline(v = metadata(bcrank)$knee, col="dodgerblue")
# legend("bottomright", legend=c("Inflection", "Knee"), bty="n", col=c("darkgreen", "dodgerblue"), lty=1, cex=1.2)
# dev.off()

w2kp = which(is.cell & e.out$Total >= metadata(bcrank)$inflection)
sce = sce[,w2kp]

write10xCounts(fname2,
               sce@assays@data$counts,
               barcodes    = colData(sce)$Barcode,
               gene.id     = rowData(sce)$ID,
               gene.symbol = rowData(sce)$Symbol,
               version     = "3",
               overwrite   = TRUE)
