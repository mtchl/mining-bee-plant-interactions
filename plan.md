# Data Mining Ecological Relationships

## Objectives

This project aims to experiment with extracting ecological relationships from existing biodiversity occurrence data. It will download images and metadata from the Atlas of Living Australia, and use a local instance of BioCLIP 2 and MegaDetector to process and annotate images. The ultimate outcome of the project will be one or more designed artefacts that visualise, explore or demonstrate the outcomes of the data mining - for example graphing or visualisation the relations uncovered.

## Stages

1. Validate the approach
    a. Identify a suitable test query / dataset for initial testing. Initial trial will be to focus on [bees](https://bie.ala.org.au/species/https%3A//biodiversity.org.au/afd/taxa/9424ef53-1dcb-42e3-95c8-7d0aafe8fba8) in the South Eastern Highlands bioregion
    b. Download and store metadata and images for this set locally
    c. Trial image processing and data-mining workflow. Initial approach:
        - use Megadetector to get the bounding box of the bee in image
        - blur the 'bee' portion of the image to redact it
        - run the processed image through BioCLIP 2 with a taxon filter set to detect plants
        - store data for species detected above a given similarity threshhold. Link these detections to the harvested occurrence data
        - evaluate success rate in identifying plant species in the bee photographs
    d. Sketch and ideate visualisation / interface design approaches 

2. Implement with multiple datasets / expanded scopes

3. Explore visualisations and design outcomes

4. Realise final outcomes

## Rules and Resources

ALA metadata is in data/SEHbees/records.csv

Images should be harvested from https://api.ala.org.au/images/image/{imageID}/large

For BioCLIP 2, use the pybioclip python API for species detection https://imageomics.github.io/pybioclip/python-api/ 

When outputting web artefacts DO NOT run browser tests in Chrome; allow the user to test in their own browser.