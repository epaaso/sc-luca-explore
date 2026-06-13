args <- commandArgs(TRUE)

if (length(args) < 1) {
    stop("Usage: Rscript run_EmptyDrop_batch.R <globalPath> [--workers N] [sample1 sample2 ...]")
}

globalPath <- normalizePath(args[1], mustWork=TRUE)
remainingArgs <- args[-1]

workers <- as.integer(Sys.getenv("EMPTYDROP_WORKERS", "1"))
workerFlag <- which(remainingArgs == "--workers")
if (length(workerFlag) > 0) {
    if (length(workerFlag) > 1 || workerFlag == length(remainingArgs)) {
        stop("Use --workers once, followed by a positive integer")
    }

    workers <- as.integer(remainingArgs[workerFlag + 1])
    remainingArgs <- remainingArgs[-c(workerFlag, workerFlag + 1)]
}

if (is.na(workers) || workers < 1) {
    stop("--workers must be a positive integer")
}

samples <- remainingArgs

if (length(samples) == 0) {
    matrixFiles <- list.files(globalPath, pattern="-matrix\\.mtx\\.gz$", full.names=FALSE)
    samples <- sort(sub("-matrix\\.mtx\\.gz$", "", matrixFiles))
}

if (length(samples) == 0) {
    stop(paste("No *-matrix.mtx.gz files found in", globalPath))
}

options(warn=1)
set.seed(100)

suppressPackageStartupMessages(library(DropletUtils))
suppressPackageStartupMessages(library(parallel))

outputRoot <- file.path(globalPath, "outputEmptyDrops")
dir.create(outputRoot, recursive=TRUE, showWarnings=FALSE)

progressFile <- file.path(outputRoot, "emptydrop_progress.tsv")
if (!file.exists(progressFile)) {
    writeLines("time\tsample\tstatus\tcells\tmessage", progressFile)
}

complete_output <- function(path) {
    all(file.exists(file.path(path, c("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"))))
}

log_progress <- function(sampleName, status, cells=NA, message="") {
    line <- paste(
        format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        sampleName,
        status,
        cells,
        gsub("[\r\n\t]", " ", message),
        sep="\t"
    )
    write(line, progressFile, append=TRUE)
}

run_sample <- function(sampleName) {
    inputPrefix <- file.path(globalPath, paste0(sampleName, "-"))
    finalPath <- file.path(outputRoot, sampleName)
    tmpPath <- file.path(outputRoot, paste0(".", sampleName, ".tmp"))

    if (complete_output(finalPath)) {
        message("[skip] ", sampleName, " already has complete output")
        log_progress(sampleName, "skipped", NA, "complete output already exists")
        return(invisible(TRUE))
    }

    if (dir.exists(finalPath)) {
        message("[clean] removing incomplete output for ", sampleName)
        unlink(finalPath, recursive=TRUE, force=TRUE)
    }

    if (dir.exists(tmpPath)) {
        message("[clean] removing stale temporary output for ", sampleName)
        unlink(tmpPath, recursive=TRUE, force=TRUE)
    }

    dir.create(tmpPath, recursive=TRUE, showWarnings=FALSE)
    if (!dir.exists(tmpPath)) {
        stop(paste("Could not create temporary output folder:", tmpPath))
    }

    message("[start] ", sampleName)
    log_progress(sampleName, "started")

    sce <- read10xCounts(inputPrefix, col.names=TRUE, type="prefix")

    if (grepl("LowerBound", sampleName, fixed=TRUE)) {
        message("[info] analysing lower-bound input for ", sampleName)
        bcrank <- barcodeRanks(counts(sce), lower=500)
    } else {
        bcrank <- barcodeRanks(counts(sce))
    }

    e.out <- emptyDrops(counts(sce))
    is.cell <- !is.na(e.out$FDR) & e.out$FDR <= 0.01
    w2kp <- which(is.cell & e.out$Total >= metadata(bcrank)$inflection)
    sce <- sce[, w2kp]

    write10xCounts(
        tmpPath,
        sce@assays@data$counts,
        barcodes=colData(sce)$Barcode,
        gene.id=rowData(sce)$ID,
        gene.symbol=rowData(sce)$Symbol,
        version="3",
        overwrite=TRUE
    )

    if (!complete_output(tmpPath)) {
        stop(paste("EmptyDrops output is incomplete for", sampleName))
    }

    if (!file.rename(tmpPath, finalPath)) {
        stop(paste("Could not move temporary output into place for", sampleName))
    }

    nCells <- length(w2kp)
    message("[done] ", sampleName, " cells=", nCells)
    log_progress(sampleName, "done", nCells)

    rm(sce, bcrank, e.out, is.cell, w2kp)
    gc()
    invisible(TRUE)
}

message("Running ", length(samples), " sample(s) with ", workers, " worker(s).")

run_sample_safely <- function(sampleName) {
    ok <- tryCatch(
        {
            run_sample(sampleName)
            TRUE
        },
        error=function(e) {
            message("[failed] ", sampleName, ": ", conditionMessage(e))
            log_progress(sampleName, "failed", NA, conditionMessage(e))
            FALSE
        }
    )

    list(sample=sampleName, ok=ok)
}

if (workers == 1) {
    results <- lapply(samples, run_sample_safely)
} else {
    results <- mclapply(
        samples,
        run_sample_safely,
        mc.cores=workers,
        mc.preschedule=FALSE
    )
}

failures <- vapply(results, function(x) if (isTRUE(x$ok)) NA_character_ else x$sample, character(1))
failures <- failures[!is.na(failures)]

if (length(failures) > 0) {
    stop(paste("EmptyDrops failed for:", paste(failures, collapse=", ")))
}

message("All requested samples are complete.")
