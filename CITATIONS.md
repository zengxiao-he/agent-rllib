# Citations and References

This document contains academic references and citations for the research and techniques used in Agent-RLlib.

## Core Papers

### Reinforcement Learning

1. **Proximal Policy Optimization Algorithms**
   ```
   @article{schulman2017proximal,
     title={Proximal policy optimization algorithms},
     author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
     journal={arXiv preprint arXiv:1707.06347},
     year={2017}
   }
   ```

2. **Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments**
   ```
   @article{lowe2017multi,
     title={Multi-agent actor-critic for mixed cooperative-competitive environments},
     author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
     journal={Advances in neural information processing systems},
     volume={30},
     year={2017}
   }
   ```

3. **Attention Is All You Need**
   ```
   @article{vaswani2017attention,
     title={Attention is all you need},
     author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
     journal={Advances in neural information processing systems},
     volume={30},
     year={2017}
   }
   ```

### LLM Integration and Tool Learning

4. **ReAct: Synergizing Reasoning and Acting in Language Models**
   ```
   @article{yao2022react,
     title={React: Synergizing reasoning and acting in language models},
     author={Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
     journal={arXiv preprint arXiv:2210.03629},
     year={2022}
   }
   ```

5. **Toolformer: Language Models Can Teach Themselves to Use Tools**
   ```
   @article{schick2023toolformer,
     title={Toolformer: Language models can teach themselves to use tools},
     author={Schick, Timo and Dwivedi-Yu, Jane and Dess{\`\i}, Roberto and Raileanu, Roberta and Lomeli, Maria and Zettlemoyer, Luke and Cancedda, Nicola and Scialom, Thomas},
     journal={arXiv preprint arXiv:2302.04761},
     year={2023}
   }
   ```

6. **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**
   ```
   @article{qin2023toolllm,
     title={Toolllm: Facilitating large language models to master 16000+ real-world apis},
     author={Qin, Yujia and Liang, Shihao and Ye, Yining and Zhu, Kunlun and Yan, Lan and Lu, Yaxi and Lin, Yankai and Cong, Xin and Tang, Xiangru and Qian, Bill and others},
     journal={arXiv preprint arXiv:2307.16789},
     year={2023}
   }
   ```

### Constitutional AI and Safety

7. **Constitutional AI: Harmlessness from AI Feedback**
   ```
   @article{bai2022constitutional,
     title={Constitutional ai: Harmlessness from ai feedback},
     author={Bai, Yuntao and Kadavath, Saurav and Kundu, Sandipan and Askell, Amanda and Kernion, Jackson and Jones, Andy and Chen, Anna and Goldie, Anna and Mirhoseini, Azalia and McKinnon, Cameron and others},
     journal={arXiv preprint arXiv:2212.08073},
     year={2022}
   }
   ```

### Curriculum Learning

8. **Curriculum Learning**
   ```
   @article{bengio2009curriculum,
     title={Curriculum learning},
     author={Bengio, Yoshua and Louradour, J{\'e}r{\^o}me and Collobert, Ronan and Weston, Jason},
     journal={Proceedings of the 26th annual international conference on machine learning},
     pages={41--48},
     year={2009}
   }
   ```

9. **Automatic Curriculum Learning for Deep RL**
   ```
   @article{portelas2020automatic,
     title={Automatic curriculum learning for deep rl: A short survey},
     author={Portelas, R{\'e}my and Colas, C{\'e}dric and Weng, Lilian and Hofmann, Katja and Oudeyer, Pierre-Yves},
     journal={arXiv preprint arXiv:2003.04664},
     year={2020}
   }
   ```

## Framework and Implementation References

### Ray RLlib

10. **Ray: A Distributed Framework for Emerging AI Applications**
    ```
    @article{moritz2018ray,
      title={Ray: A distributed framework for emerging {AI} applications},
      author={Moritz, Philipp and Nishihara, Robert and Wang, Stephanie and Tumanov, Alexey and Liaw, Richard and Liang, Eric and Elibol, Melih and Yang, Zongheng and Paul, William and Jordan, Michael I and others},
      journal={13th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 18)},
      pages={561--577},
      year={2018}
    }
    ```

### PyTorch

11. **PyTorch: An Imperative Style, High-Performance Deep Learning Library**
    ```
    @article{paszke2019pytorch,
      title={PyTorch: An imperative style, high-performance deep learning library},
      author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
      journal={Advances in neural information processing systems},
      volume={32},
      year={2019}
    }
    ```

## Application Domain References

### Conversational AI

12. **BlenderBot: Recipes for Building an Open-Domain Chatbot**
    ```
    @article{roller2020recipes,
      title={Recipes for building an open-domain chatbot},
      author={Roller, Stephen and Dinan, Emily and Goyal, Naman and Ju, Da and Williamson, Mary and Liu, Yinhan and Xu, Jing and Ott, Myle and Shuster, Kurt and Smith, Eric M and others},
      journal={arXiv preprint arXiv:2004.13637},
      year={2020}
    }
    ```

13. **LaMDA: Language Models for Dialog Applications**
    ```
    @article{thoppilan2022lamda,
      title={LaMDA: Language Models for Dialog Applications},
      author={Thoppilan, Romal and De Freitas, Daniel and Hall, Jamie and Shazeer, Noam and Kulshreshtha, Apoorv and Cheng, Heng-Tze and Jin, Alicia and Bos, Taylor and Baker, Leslie and Du, Yu and others},
      journal={arXiv preprint arXiv:2201.08239},
      year={2022}
    }
    ```

### Multi-Agent Systems

14. **Emergent Communication in Multi-Agent Reinforcement Learning**
    ```
    @article{foerster2018emergent,
      title={Emergent communication in multi-agent reinforcement learning},
      author={Foerster, Jakob and Assael, Yannis M and de Freitas, Nando and Whiteson, Shimon},
      journal={Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems},
      pages={537--545},
      year={2018}
    }
    ```

## Evaluation and Benchmarking

15. **Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models**
    ```
    @article{srivastava2022beyond,
      title={Beyond the imitation game: Quantifying and extrapolating the capabilities of language models},
      author={Srivastava, Aarohi and Rastogi, Abhinav and Rao, Abhishek and Shoeb, Abu Awal Md and Abid, Abubakar and Fisch, Adam and Brown, Adam R and Santoro, Adam and Gupta, Aditya and Garriga-Alonso, Adri{\`a} and others},
      journal={arXiv preprint arXiv:2206.04615},
      year={2022}
    }
    ```

## Related Work and Inspiration

### OpenAI GPT Series

16. **Language Models are Few-Shot Learners**
    ```
    @article{brown2020language,
      title={Language models are few-shot learners},
      author={Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and others},
      journal={Advances in neural information processing systems},
      volume={33},
      pages={1877--1901},
      year={2020}
    }
    ```

### Anthropic Claude

17. **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**
    ```
    @article{bai2022training,
      title={Training a helpful and harmless assistant with reinforcement learning from human feedback},
      author={Bai, Yuntao and Jones, Andy and Ndousse, Kamal and Askell, Amanda and Chen, Anna and DasSarma, Nova and Drain, Dawn and Fort, Stanislav and Ganguli, Deep and Henighan, Tom and others},
      journal={arXiv preprint arXiv:2204.05862},
      year={2022}
    }
    ```

## Acknowledgments

We acknowledge the contributions of the open-source community and the researchers whose work has made this project possible. Special thanks to:

- The Ray team for the excellent RLlib framework
- OpenAI and Anthropic for advancing LLM research
- The PyTorch team for the deep learning framework
- The broader reinforcement learning and NLP research communities

## How to Cite This Work

If you use Agent-RLlib in your research, please cite:

```
@software{agent_rllib_2023,
  title={Agent-RLlib: Multi-Agent Reinforcement Learning with LLM Integration},
  author={Your Name},
  year={2023},
  url={https://github.com/yourusername/agent-rllib},
  version={0.2.1}
}
```

---

*This citation file is regularly updated as new relevant research emerges and as the project evolves.*
