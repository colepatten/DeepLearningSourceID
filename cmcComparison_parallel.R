library(cmcR)
library(x3ptools)
library(ggplot2)
library(magrittr)
library(dplyr)
library(foreach)
library(doParallel)
library(filelock)


n_cores <- detectCores()
cluster <- makeCluster(n_cores - 1)
registerDoParallel(cluster)

processed_path <- "processed_x3p_path"
files <- list.files(path=processed_path, pattern = "\\.x3p$")
excel_path <- "info_csv_path"
log_path <- "progress.log path to log the progress of cmc"



# import x3p files
x3ps <- list()
for (i in 1:length(files)){
  cur_file = files[i]
  im <- x3ptools::x3p_read(paste0(processed_path, cur_file))
  x3ps <- append(x3ps, list(im))
  names(x3ps)[i] <- cur_file
}

#import data regarding x3p files
dat <- read.csv(excel_path, head=TRUE)
print("Files Read \n")


check_headers <- function(x3p) {
  list(incrementX = x3p$header.info$incrementX,
       incrementY = x3p$header.info$incrementY)
}

header_info <- lapply(x3ps[1:5], check_headers)
print(header_info)


# Function to log progress
log_progress <- function(message) {
  lock <- lockfile::lock(filelock::lock(log_path))
  cat(message, file = log_path, append = TRUE, sep = "\n")
  lockfile::unlock(lock)
}



results <- foreach(i=1:1, .combine='rbind', .packages = c('dplyr', 'purrr', 'cmcR')) %:%
    foreach(j=(i+1):3, .combine='rbind') %dopar% {
    
        message <- sprintf("Comparing i: %d and j: %d\n", i, j)
        log_progress(message)
        
        #select images to compare
        index1 = dat$File[i]
        index2 = dat$File[j]
        
        print(check_headers(x3ps[[index1]]))
        print(check_headers(x3ps[[index2]]))
        
        #forward compare
        kmComparisonFeatures <- purrr::map_dfr(seq(-180,180,by = 3),
                                               ~ cmcR::comparison_allTogether(reference = x3ps[[index1]],
                                                                        target = x3ps[[index2]],
                                                                        theta = .,
                                                                        returnX3Ps = TRUE))
        #forward decision
        kmComparison_allCMCs <- kmComparisonFeatures %>%
          mutate(originalMethodClassif = cmcR::decision_CMC(cellIndex = cellIndex,
                                                      x = x,
                                                      y = y,
                                                      theta = theta,
                                                      corr = pairwiseCompCor,
                                                      xThresh = 40,
                                                      yThresh = 40,
                                                      thetaThresh = 10,
                                                      corrThresh = .5),
                 highCMCClassif = cmcR::decision_CMC(cellIndex = cellIndex,
                                               x = x,
                                               y = y,
                                               theta = theta,
                                               corr = pairwiseCompCor,
                                               xThresh = 40,
                                               yThresh = 40,
                                               thetaThresh = 10,
                                               corrThresh = .5,
                                               tau = 1))
        #backward compare
        kmComparisonFeatures_rev <- purrr::map_dfr(seq(-180,180,by = 3),
                                                   ~ cmcR::comparison_allTogether(reference = x3ps[[index2]],
                                                                            target = x3ps[[index1]],
                                                                            theta = .,
                                                                            returnX3Ps = TRUE))
        #backward decision
        kmComparison_allCMCs_rev <- kmComparisonFeatures_rev %>%
          mutate(originalMethodClassif = cmcR::decision_CMC(cellIndex = cellIndex,
                                                      x = x,
                                                      y = y,
                                                      theta = theta,
                                                      corr = pairwiseCompCor,
                                                      xThresh = 40,
                                                      thetaThresh = 10,
                                                      corrThresh = .5),
                 highCMCClassif = cmcR::decision_CMC(cellIndex = cellIndex,
                                               x = x,
                                               y = y,
                                               theta = theta,
                                               corr = pairwiseCompCor,
                                               xThresh = 40,
                                               thetaThresh = 10,
                                               corrThresh = .5,
                                               tau = 1))
        #combine decision directions
        if (kmComparison_allCMCs$highCMCClassif[1] == "non-CMC (failed)" ||
            kmComparison_allCMCs_rev$highCMCClassif[1] == "non-CMC (failed)"){
          
          forward_cmcs = sum(kmComparison_allCMCs$originalMethodClassif == 'CMC')
          backward_cmcs = sum(kmComparison_allCMCs_rev$originalMethodClassif == 'CMC')
          cmcMethod = "original"
          
        } else {
        
          forward_cmcs = sum(kmComparison_allCMCs$highCMCClassif == 'CMC')
          backward_cmcs = sum(kmComparison_allCMCs_rev$highCMCClassif == 'CMC')
          total_cmcs = forward_cmcs+backward_cmcs
          cmcMethod = "highCMC"
        }
        
        
        #check if match
        if (dat$Firearm[dat$File == index1] == dat$Firearm[dat$File == index2]){
          match = 1
        } else{
          match = 0
        }
        print(paste(cmcMethod[i,j], cmcMethod[j,i], cmcCounts[i,j]+cmcCounts[i,j]))
        c(i, j, forward_cmcs, backward_cmcs, cmcMethod, match)
    }


write.csv(results, file="results_path")
stopCluster(cl = cluster)
