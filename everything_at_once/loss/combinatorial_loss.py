import torch.nn as nn

from everything_at_once.loss.contrastive_losses import NormSoftmaxLoss, MMS_Loss
from everything_at_once.model.utils.utils import sim_matrix


class CombinatorialLoss(nn.Module):
    def __init__(self, contrastive_loss='NormSoftmax', temperature=0.02,
                 tv_weight=0, ta_weight=0, va_weight=0,
                 t_va_weight=0, v_ta_weight=0, a_tv_weight=0):
        super().__init__()

        if contrastive_loss == 'NormSoftmax':
            self.contrastive_loss = NormSoftmaxLoss(temperature=temperature)
        elif contrastive_loss == 'MMS':
            self.contrastive_loss = MMS_Loss()
        else:
            raise NotImplementedError()

        self.tv_weight = tv_weight
        self.vt_weight = 0

        self.ta_weight = ta_weight
        self.at_weight = 0
        
        self.va_weight = va_weight
        self.av_weight = 0
        
        self.t_va_weight = t_va_weight
        self.va_t_weight = 1
        
        self.v_ta_weight = v_ta_weight
        self.ta_v_weight = 1

        self.a_tv_weight = a_tv_weight
        self.tv_a_weight = 1

    def forward(self, input_data):

        nonempty = {}
        #print('input_data.keys():', input_data.keys())
        #print("shape of first tensor",input_data['text_nonempty_input_mask'])
        #print("shape of second tensor",input_data['video_nonempty_input_mask'])
        #print("shape of tv",input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask'])


        nonempty['tv'] = input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask']
        nonempty['vt'] = input_data['video_nonempty_input_mask'] & input_data['text_nonempty_input_mask']

        nonempty['ta'] = input_data['text_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']
        nonempty['at'] = input_data['audio_nonempty_input_mask'] & input_data['text_nonempty_input_mask']

        nonempty['va'] = input_data['video_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']
        nonempty['av'] = input_data['audio_nonempty_input_mask']& input_data['video_nonempty_input_mask']
        """
        nonempty['t_va'] = input_data['text_nonempty_input_mask'] & (
                    input_data['video_nonempty_input_mask'] & input_data['audio_nonempty_input_mask'])
        nonempty['va_t'] = (input_data['video_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']) & input_data['text_nonempty_input_mask']
        
        nonempty['v_ta'] = input_data['video_nonempty_input_mask'] & (
                    input_data['text_nonempty_input_mask'] & input_data['audio_nonempty_input_mask'])
        nonempty['ta_v'] = (input_data['text_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']) & input_data['video_nonempty_input_mask']
        
        nonempty['a_tv'] = input_data['audio_nonempty_input_mask'] & (
                    input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask'])
        
        nonempty['tv_a'] = (input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask']) & input_data['audio_nonempty_input_mask']
        """
        loss_sum = 0
        weight_sum = 0
        loss_info = {}

        for name, embed_name1, embed_name2, weight in [
            ('vt','video_embed','text_embed',self.vt_weight),
            ('tv', 'text_embed', 'video_embed', self.tv_weight),
            ('at', 'audio_embed','text_embed', self.ta_weight), # fix this
            ('ta', 'text_embed', 'audio_embed', self.at_weight), 
            ('va', 'video_embed', 'audio_embed', self.va_weight),
            ('av','audio_embed', 'video_embed', self.av_weight),
            #('va_t','va_embed','text_embed',self.va_t_weight),
            #('t_va', 'text_embed', 'va_embed', self.t_va_weight),
            #('ta_v','ta_embed','video_embed',self.ta_v_weight),
            #('v_ta', 'video_embed', 'ta_embed', self.v_ta_weight),
            #('a_tv', 'audio_embed', 'tv_embed', self.a_tv_weight),
            #('tv_a','tv_embed','audio_embed',self.tv_a_weight),
        ]:
            if (embed_name1 in input_data) and (embed_name2 in input_data) and (weight != 0):
                nonempty_mask = nonempty[name]
                #print("nonempty shape",nonempty_mask.shape)
                #print("embed 1 before",input_data[embed_name1].shape)
                #print("embed 2 before",input_data[embed_name2].shape)
                embed1 = input_data[embed_name1][nonempty_mask]
                embed2 = input_data[embed_name2][nonempty_mask]
                #print("embed 1",embed1.shape)
                #print("embed 2",embed2.shape)

                loss = self.contrastive_loss(sim_matrix(embed1, embed2))
                loss_info[name] = loss.item()
                loss_sum += weight * loss
                weight_sum += weight

        final_loss = loss_sum / weight_sum
        loss_info['Retrieval'] = final_loss.item()
        return final_loss, loss_info
