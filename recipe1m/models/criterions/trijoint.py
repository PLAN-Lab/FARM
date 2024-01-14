import torch
import torch.nn as nn
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from .triplet import Triplet
from .pairwise import Pairwise
from .logRatio import LogRatio
import random
 
class Trijoint(nn.Module):

    def __init__(self, opt, nb_classes, dim_emb, with_classif=False, engine=None, tmp=None, with_fga=False, with_fga_img=False):
        super(Trijoint, self).__init__()
        
        self.with_classif = with_classif
        self.with_fga = opt['with_fga']
        self.with_fga_img = opt['with_fga_img']
        self.with_fga_recipe = opt['with_fga_recipe']
        self.with_fga_random = opt['with_fga_random']
        self.fga_random_threshold = opt['fga_random_threshold']
        self.with_hyperbolic = opt['with_hyperbolic']
        self.hyperbolic_loss_weight = opt['hyperbolic_loss_weight']
        
        self.fga_pairs = [] 
        
        if self.with_fga_recipe:
            self.fga_pairs += [
            ('recipe_embedding', 'title_embedding', 'recipe', 'recipe'),
            ('recipe_embedding', 'ingr_embedding', 'recipe', 'recipe'),
            ('recipe_embedding', 'instr_embedding', 'recipe', 'recipe'),
            ('title_embedding', 'recipe_embedding', 'recipe', 'recipe'),
            ('ingr_embedding', 'recipe_embedding', 'recipe', 'recipe'),
            ('instr_embedding', 'recipe_embedding', 'recipe', 'recipe'),
        ]

        if self.with_fga_img:
            self.fga_pairs += [
                ('image_embedding', 'title_embedding', 'image', 'recipe'),
                ('image_embedding', 'ingr_embedding', 'image', 'recipe'),
                ('image_embedding', 'instr_embedding', 'image', 'recipe'),
                ('title_embedding', 'image_embedding', 'recipe', 'image'),
                ('ingr_embedding', 'image_embedding', 'recipe', 'image'),
                ('instr_embedding', 'image_embedding', 'recipe', 'image'),
            ]

        self.hyperbolic_pairs = [
            ('recipe_embedding', 'recipe'),
            ('title_embedding', 'recipe'),
            ('ingr_embedding', 'recipe'),
            ('instr_embedding', 'recipe'),
            ('image_embedding', 'image')
                ]

        if self.with_classif:
            self.weight_classif = opt['weight_classif']
            if self.weight_classif == 0:
                Logger()('You should use "--model.with_classif False"', Logger.ERROR)
            self.weight_retrieval = 1 - 2 * opt['weight_classif']

        self.keep_background = opt.get('keep_background', False)
        if self.keep_background:
            # http://pytorch.org/docs/master/nn.html?highlight=crossentropy#torch.nn.CrossEntropyLoss
            self.ignore_index = -100
        else:
            self.ignore_index = 0

        Logger()('ignore_index={}'.format(self.ignore_index))
        if self.with_classif:
            self.criterion_image_classif = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            self.criterion_recipe_classif = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.retrieval_strategy = opt['retrieval_strategy.name']

        if self.retrieval_strategy == 'triplet':
            self.criterion_retrieval = Triplet(
                opt,
                nb_classes,
                dim_emb,
                engine)
            
            self.criterion_retrieval_fga = Triplet(
                opt,
                nb_classes,
                dim_emb,
                engine)

        elif self.retrieval_strategy == 'pairwise':
            self.criterion_retrieval = Pairwise(opt)

        elif self.retrieval_strategy == 'pairwise_pytorch':
            self.criterion_retrieval = nn.CosineEmbeddingLoss()
        else:
            raise ValueError('Unknown loss ({})'.format(self.retrieval_strategy))

        self.cross_encoder = opt.get('cross_encoder', False)
        
        if self.cross_encoder:
            self.cross_criterion = nn.CrossEntropyLoss()
            self.itm_loss_weight = opt.get('itm_loss_weight', 1)

    def forward(self, net_out, batch):
        #print(net_out['image_embedding'].shape)
        #print(net_out['recipe_embedding'].shape)
        #print(net_out['title_embedding'].shape)
        #print(net_out['ingr_embedding'].shape)
        #print(net_out['instr_embedding'].shape)
        #print(batch['match'].shape)
        #print(batch['image']['class_id'].shape)
        #print(batch['recipe']['class_id'].shape)
        
        if self.retrieval_strategy in ['triplet']:
            out = None
            random_number = random.random()
            out = self.criterion_retrieval(net_out['image_embedding'],
                                            net_out['recipe_embedding'],
                                            batch['match'],
                                            batch['image']['class_id'],
                                            batch['recipe']['class_id'])
                
            if self.with_fga:
                if self.with_fga_random and random_number<=self.fga_random_threshold:
                    out['loss']  = 0
                    add_fga = True
                elif self.with_fga_random and random_number>self.fga_random_threshold:
                    add_fga = False
                else:
                    add_fga = True
                if add_fga:
                    fga_loss = 0
                    for fga_pair in self.fga_pairs:
                        fga_out = self.criterion_retrieval(net_out[fga_pair[0]],
                                            net_out[fga_pair[1]],
                                            batch['match'],
                                            batch[fga_pair[2]]['class_id'],
                                            batch[fga_pair[3]]['class_id'])
                        fga_loss += fga_out['loss']

                    
                    #with open("loss_fga.out", "a") as f:
                    #    f.write(str(fga_loss) + "," + str(out['loss']) + '\n')
                    out['loss'] += (fga_loss/len(self.fga_pairs))

            if self.with_hyperbolic and len(batch['image']['class_id'].unique()) > 1 and len(batch['recipe']['class_id'].unique()) > 1:
                log_ratio_loss = LogRatio()
                max_hyperbolic = 0
                hyperbolic_loss = 0
                cnt = 0
                for x in self.hyperbolic_pairs:
                    cur_loss = log_ratio_loss( net_out[x[0]], batch[x[1]]['class_id'] )
                    if torch.isnan(cur_loss):
                        continue
                    if cur_loss > max_hyperbolic:
                        max_hyperbolic = cur_loss
                    hyperbolic_loss += cur_loss
                    cnt += 1
                if max_hyperbolic != 0:
                    hyperbolic_loss = hyperbolic_loss / max_hyperbolic
                #hyperbolic_losses = torch.stack([log_ratio_loss( net_out[x[0]], batch[x[1]]['class_id'] )[0] for x in self.hyperbolic_pairs])
                #hyperbolic_loss = (hyperbolic_losses / hyperbolic_losses.max() / len(hyperbolic_losses)).sum()
                out['loss'] += (hyperbolic_loss * self.hyperbolic_loss_weight)
                #with open('log_r.out', 'a') as f:
                #    f.write('yes ' + str(cnt) + '\n')
                
                del hyperbolic_loss, log_ratio_loss
                torch.cuda.empty_cache()
            else:
                #with open('log_r.out', 'a') as f:
                #    f.write('no\n')
                pass

        elif self.retrieval_strategy == 'pairwise':
            out = self.criterion_retrieval(net_out['image_embedding'],
                                           net_out['recipe_embedding'],
                                           batch['match'])
        elif self.retrieval_strategy == 'pairwise_pytorch':
            out = {}
            out['loss'] = self.criterion_retrieval(net_out['image_embedding'],
                                                   net_out['recipe_embedding'],
                                                   batch['match'])

        if self.with_classif:
            out['image_classif'] = self.criterion_image_classif(net_out['image_classif'],
                                                                batch['image']['class_id'].squeeze(1))
            out['recipe_classif'] = self.criterion_recipe_classif(net_out['recipe_classif'],
                                                                  batch['recipe']['class_id'].squeeze(1))
            out['loss'] *= self.weight_retrieval
            out['loss'] += out['image_classif'] * self.weight_classif
            out['loss'] += out['recipe_classif'] * self.weight_classif

        if self.cross_encoder and self.itm_loss_weight > 0:
            out['loss']+= self.cross_criterion(net_out['cross_embedding'], net_out['cross_embedding_labels'])*self.itm_loss_weight
            
        torch.cuda.empty_cache()
        
        return out
