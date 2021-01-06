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

