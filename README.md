# TrajectoryFor-and-DPP
We mix Determinantal Point Processes with Transformers for Trajectory Forecasting on ETH/UCY. Final project in the course of Advanced Machine Learning at Sapienza Data Science's Master's degree.

![Screenshot](TFDPPff.jpg)
## Authors:
-Leonardo Placidi

-Negin Amininodoushan




# How to run

To run any code just use Google Colab, download the [repository](https://github.com/FGiuliari/Trajectory-Transformer/) of Prof.Galasso(since models are based on that, you can copy it writing !git clone https://github.com/FGiuliari/Trajectory-Transformer/), add the files of this repository in the directory and write:

cd '/directorywhereyouarerunningeverything'

CUDA_VISIBLE_DEVICES=0

!python NAMEOFFILE.py --dataset_name dataset(or zara1 or zara2) --name dataset(or zara1 or zara2) --batch_size xnumx

there are many other customizations of the model and are the same of the Transformers by Prof.Galasso from the [repository](https://github.com/FGiuliari/Trajectory-Transformer/) .

# Files
First of all, we started to code from the Transformers repository by FrancescoGiuliari4, and you can find our work in the repository by Gruntrexpewrus5. Thefiles that we produce are the following, all but the last are about the QuantizedTransformer with DPP, the last is about the Transformer coupled with the DPPsampling after output(that was the best method):

•**QuantizedTFsamples.py**: this originally was the main class of Prof.GalassoTransformer, we introduced here, after the decoder, a sampling of from theouput. The class is able to return both the cluster chosen and the relativeposition.

•**LossLeNeg.py**:  Here is implemented the custom Loss following the algorithm for the DSF(γ) training of Kitani’s paper, but adapted by us for theTransformer. More detail will come below.

•**SamplerLeNeg.py**:  This is the module where,  using torch.multinomial,we sample at each step some clusters from the 1000 dimensional vectorreturned by the standard quantized transformer.

•**TESTDPP.py**:  Here we coded all the metrics, defined a recursive functionto create the mutlifutures and tested our results.

•**test_quantizedTF.py**: this is the code for the test using Prof.Galasso Transformer, we implemented also here our metrics and functions to compare theresults later with our new method.

•**DPPsampler.py**:  this is the code for the test using Prof.Galasso Transformer and implement DPP sampler on the result of it to add diversity.

# Bibliography
As stated above.
Our codes and models start from:

@misc{giuliari2020transformer,
      title={Transformer Networks for Trajectory Forecasting}, 
      author={Francesco Giuliari and Irtiza Hasan and Marco Cristani and Fabio Galasso},
      year={2020},
      eprint={2003.08111},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

and we based the DPP implementation from the paper:

@misc{yuan2019diverse,
      title={Diverse Trajectory Forecasting with Determinantal Point Processes}, 
      author={Ye Yuan and Kris Kitani},
      year={2019},
      eprint={1907.04967},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{Kulesza_2012,
   title={Determinantal Point Processes for Machine Learning},
   volume={5},
   ISSN={1935-8245},
   url={http:// dx.doi.org/10.1561/2200000044},
   DOI={10.1561/2200000044},
   number={2-3},
   journal={Foundations and Trends® in Machine Learning},
   publisher={Now Publishers},
   author={Kulesza, Alex},
   year={2012},
   pages={123–286}
}

## Thanks to Professor Fabio Galasso for giving us a challenging last project and thanks to Dr.Luca Franco for the helpful suggestions and discussions.
