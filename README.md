<div  align="center">    
 <img src="resources/datasets4eo.png" width = "400" height = "130" alt="segmentation" align=center />
</div>


![example workflow](https://github.com/github/docs/actions/workflows/main.yml/badge.svg)

[Homepage of the project](https://earthnets.nicepage.io/)

# Dataset4EO: A Python Library for Efficient Dataset Management in Earth Observation

Dataset4EO is a Python library designed to streamline the creation, storage, and benchmarking of Earth observation datasets. The library focuses on two primary methods of handling large multi-channel remote sensing data:

1. **Channel-Wise Storage**: Stores each channel of a multi-channel image as an independent chunk, allowing for selective decoding of specific channels.
2. **Full-Image Storage**: Stores entire multi-channel images in chunks and selects specific channels during the decoding phase.

---

## **Key Features**

- **Channel-Wise Dataset Support**: Efficient storage and selective decoding of individual image channels.
- **Full-Image Dataset Support**: Traditional storage and decoding of entire multi-channel images.
- **Performance Benchmarking**: Tools to compare storage efficiency, memory usage, and decoding speed between channel-wise and full-image approaches.
- **Integration with LitData**: Fully leverages LitData's streaming capabilities for handling large datasets.

---

## **Installation**

```bash
pip install -e .
```

Todo List

- [ ] Re-organize mroe than 400 datasets in Remote sensing cimmunity in a task-oriented way;
- [ ] supporting for heigh-level repos for specific tasks: obejct detection, segmentation and so forth;
- [ ] supporting dataloaders in a easy-to-use way for custom projects;

## Supported datasets:

- [ ] [DFC2020](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest)
- [ ] [LandSlide4Sense](https://www.iarai.ac.at/landslide4sense/)
- [ ] [Eurosat](https://github.com/phelber/EuroSAT#)
- [ ] [AID](https://captain-whu.github.io/AID/)
- [ ] [DIOR](http://www.escience.cn/people/JunweiHan/DIOR.html)
- [ ] [DOTA 2.0](https://captain-whu.github.io/DOTA/index.html)
- [ ] [fMoW](https://github.com/fMoW/dataset)
- [ ] [GeoNRW](https://github.com/gbaier/geonrw)
- [ ] [LoveDA](https://github.com/Junjue-Wang/LoveDA)
- [ ] [NWPU_VHR10](https://github.com/chaozhong2010/VHR-10_dataset_coco)
- [ ] [RSUSS](https://github.com/EarthNets/RSI-MMSegmentation)
- [ ] [BigEarthNet](https://bigearth.net/)
- [ ] [SEASONET](https://zenodo.org/record/5850307#.Y0cayXbP1D8)
- [ ] [SSL4EO_S12](https://github.com/zhu-xlab/SSL4EO-S12)
- [ ] [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
- [ ] [Satlas](https://satlas-pretrain.allen.ai)


On-going...