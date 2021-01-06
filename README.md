# TrajectoryFor-and-DPP
We mix Determinantal Point Processes with Transformers for Trajectory Forecasting on ETH/UCY

To run any code just use Google Colab and write:
cd '/content/Trajectory-Transformer'

CUDA_VISIBLE_DEVICES=0
!python NAMEOFFILE.py --dataset_name dataset(or zara1 or zara2) --name dataset(or zara1 or zara2) --batch_size xnumx
there are many other customizations of the model and are the same of the Transformers by Prof.Galasso from the repository: https://github.com/FGiuliari/Trajectory-Transformer/.

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

We produced:

\begin{itemize}
    \item \textbf{Quantized\_TFsamples.py}: this originally was the main class of Prof.Galasso Transformer, we introduced here, after the decoder, a sampling of from the ouput. The class is able to return both the cluster chosen and the relative position.
    \item \textbf{LossLeNeg.py}: Here is implemented the custom Loss following the algorithm for the DSF($\gamma$) training of Kitani's paper, but adapted by us for the Transformer. More detail will come below.
    \item \textbf{Sampler\_LeNeg.py}: This is the module where, using torch.multinomial, we sample at each step some clusters from the 1000 dimensional vector returned by the standard quantized transformer.
    \item \textbf{TESTDPP.py}: Here we coded all the metrics, defined a recursive function to create the mutlifutures and tested our results.
    \item \textbf{test\_quantizedTF.py}: this is the code for the test using Prof.Galasso Transformer, we implemented also here our metrics and functions to compare the results later with our new method.
    \item \textbf{DPP\_sampler.py}: this is the code for the test using Prof.Galasso Transformer and implement DPP sampler on the result of it to add diversity.
\end{itemize}
