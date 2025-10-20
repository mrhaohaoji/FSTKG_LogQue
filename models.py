from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

pi = 3.14159265358979323846

def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y


def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y


class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale

class LogQFSTKG(nn.Module):
    def __init__(self, nentity, nrelation, ntime, nlocation, nfuzziness, hidden_dim, gamma, test_batch_size=1, use_cuda=False, query_name_dict=None, center_reg=None, drop=0.):
        super(LogQFSTKG, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.ntime = ntime
        self.nlocation = nlocation
        self.nfuzziness = nfuzziness
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(
            nentity).to(torch.float).repeat(test_batch_size, 1) 
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.time_dim = hidden_dim
        self.location_dim = hidden_dim
        self.fuzziness_dim = hidden_dim

        self.cen = center_reg

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim), requires_grad=True)

        self.angle_scale = AngleScale(self.embedding_range.item())

        self.modulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)

        self.axis_scale = 1.0
        self.arg_scale = 1.0

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.axis_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.axis_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.arg_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.arg_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.axis_time_embedding = nn.Parameter(torch.zeros(ntime, self.time_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.axis_time_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.arg_time_embedding = nn.Parameter(torch.zeros(ntime, self.time_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.arg_time_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.axis_location_embedding = nn.Parameter(torch.zeros(nlocation, self.location_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.axis_location_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.arg_location_embedding = nn.Parameter(torch.zeros(nlocation, self.location_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.arg_location_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.axis_fuzziness_embedding = nn.Parameter(torch.zeros(nfuzziness, self.fuzziness_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.axis_fuzziness_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.arg_fuzziness_embedding = nn.Parameter(torch.zeros(nfuzziness, self.fuzziness_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.arg_fuzziness_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.cone_proj = ConeProjection(self.entity_dim, 1600, 2)
        self.cone_intersection = ConeIntersection(self.entity_dim, drop)
        self.cone_negation = ConeNegation()
        self.cone_after = ConeAfter()
        self.cone_before = ConeBefore()

    def transform_union_query(self, queries, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF' \
                or self.query_name_dict[query_structure] == '2u_t-DNF' \
                or self.query_name_dict[query_structure] =='2u_l-DNF' \
                or self.query_name_dict[query_structure] == '2u_f-DNF':
            queries = queries[:, :-1]
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :5], queries[:, 11:]], dim=1),
                                 torch.cat([queries[:, 5:10], queries[:, 11:]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return 'e', (('r', 't', 'l', 'f'),)
        elif self.query_name_dict[query_structure] == '2u_t-DNF':
            return 'e', (('r', 'e', 'l', 'f'),)
        elif self.query_name_dict[query_structure] == '2u_l-DNF':
            return 'e', (('r', 'e', 't', 'f'),)
        elif self.query_name_dict[query_structure] == '2u_f-DNF':
            return 'e', (('r', 'e', 't', 'l'),)

        elif self.query_name_dict[query_structure] == 'up-DNF':
            return 'e', (('r', 't', 'l', 'f'), ('r', 't', 'l', 'f'))

    def train_step(self, model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    def embed_query_cone_time(self, queries, query_structure, idx):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in [('r', 'e', 'l', 'f'), 'n', 'a', 'b']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(axis_entity_embedding)
                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding)
                idx += 1

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding
            else:
                axis_embedding, arg_embedding, idx = self.embed_query_cone_time(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    axis_embedding, arg_embedding = self.cone_negation(axis_embedding, arg_embedding)
                elif query_structure[-1][i] == 'a':
                    assert (queries[:, idx] == -3).all()
                    axis_embedding, arg_embedding = self.cone_after(axis_embedding, arg_embedding)
                elif query_structure[-1][i] == 'b':
                    assert (queries[:, idx] == -4).all()
                    axis_embedding, arg_embedding = self.cone_before(axis_embedding, arg_embedding)

                else:
                    axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, idx])
                    arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, idx])
                    axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                    arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)
                    axis_r_embedding = convert_to_axis(axis_r_embedding)
                    arg_r_embedding = convert_to_axis(arg_r_embedding)

                    axis_entity2_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx+1])
                    axis_entity2_embedding = self.angle_scale(axis_entity2_embedding, self.axis_scale)
                    axis_entity2_embedding = convert_to_axis(axis_entity2_embedding)
                    if self.use_cuda:
                        arg_entity2_embedding = torch.zeros_like(axis_entity2_embedding).cuda()
                    else:
                        arg_entity2_embedding = torch.zeros_like(axis_entity2_embedding)

                    axis_location_embedding = torch.index_select(self.axis_location_embedding, dim=0, index=queries[:, idx + 2])
                    arg_location_embedding = torch.index_select(self.arg_location_embedding, dim=0, index=queries[:, idx + 2])
                    axis_location_embedding = self.angle_scale(axis_location_embedding, self.axis_scale)
                    arg_location_embedding = self.angle_scale(arg_location_embedding, self.arg_scale)
                    axis_location_embedding = convert_to_axis(axis_location_embedding)
                    arg_location_embedding = convert_to_axis(arg_location_embedding)

                    axis_fuzziness_embedding = torch.index_select(self.axis_fuzziness_embedding, dim=0, index=queries[:, idx + 3])
                    arg_fuzziness_embedding = torch.index_select(self.arg_fuzziness_embedding, dim=0, index=queries[:, idx + 3])
                    axis_fuzziness_embedding = self.angle_scale(axis_fuzziness_embedding, self.axis_scale)
                    arg_fuzziness_embedding = self.angle_scale(arg_fuzziness_embedding, self.arg_scale)
                    axis_fuzziness_embedding = convert_to_axis(axis_fuzziness_embedding)
                    arg_fuzziness_embedding = convert_to_axis(arg_fuzziness_embedding)

                    axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding,
                                                                   axis_r_embedding + axis_entity2_embedding + axis_location_embedding + axis_fuzziness_embedding,
                                                                   arg_r_embedding + arg_entity2_embedding + arg_location_embedding + arg_fuzziness_embedding)
                if query_structure[-1][i] == 'n' or query_structure[-1][i] == 'a' or query_structure[-1][i] == 'b':
                    idx += 1
                else:
                    idx += 4
        else:
            axis_embedding_list = []
            arg_embedding_list = []
            for i in range(len(query_structure)):
                axis_embedding, arg_embedding, idx = self.embed_query_cone_time(queries, query_structure[i], idx)
                axis_embedding_list.append(axis_embedding)
                arg_embedding_list.append(arg_embedding)

            stacked_axis_embeddings = torch.stack(axis_embedding_list)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)

            axis_embedding, arg_embedding = self.cone_intersection(stacked_axis_embeddings, stacked_arg_embeddings)

        return axis_embedding, arg_embedding, idx

    def embed_query_cone_location(self, queries, query_structure, idx):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in [('r', 'e', 't', 'f'), 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(axis_entity_embedding)
                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding)
                idx += 1

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding
            else:
                axis_embedding, arg_embedding, idx = self.embed_query_cone_location(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    axis_embedding, arg_embedding = self.cone_negation(axis_embedding, arg_embedding)

                else:
                    axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, idx])
                    arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, idx])
                    axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                    arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)
                    axis_r_embedding = convert_to_axis(axis_r_embedding)
                    arg_r_embedding = convert_to_axis(arg_r_embedding)

                    axis_entity2_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx+1])
                    axis_entity2_embedding = self.angle_scale(axis_entity2_embedding, self.axis_scale)
                    axis_entity2_embedding = convert_to_axis(axis_entity2_embedding)
                    if self.use_cuda:
                        arg_entity2_embedding = torch.zeros_like(axis_entity2_embedding).cuda()
                    else:
                        arg_entity2_embedding = torch.zeros_like(axis_entity2_embedding)

                    axis_time_embedding = torch.index_select(self.axis_time_embedding, dim=0, index=queries[:, idx+2])
                    arg_time_embedding = torch.index_select(self.arg_time_embedding, dim=0, index=queries[:, idx+2])
                    axis_time_embedding = self.angle_scale(axis_time_embedding, self.axis_scale)
                    arg_time_embedding = self.angle_scale(arg_time_embedding, self.arg_scale)
                    axis_time_embedding = convert_to_axis(axis_time_embedding)
                    arg_time_embedding = convert_to_axis(arg_time_embedding)

                    axis_fuzziness_embedding = torch.index_select(self.axis_fuzziness_embedding, dim=0, index=queries[:, idx+3])
                    arg_fuzziness_embedding = torch.index_select(self.arg_fuzziness_embedding, dim=0, index=queries[:, idx+3])
                    axis_fuzziness_embedding = self.angle_scale(axis_fuzziness_embedding, self.axis_scale)
                    arg_fuzziness_embedding = self.angle_scale(arg_fuzziness_embedding, self.arg_scale)
                    axis_fuzziness_embedding = convert_to_axis(axis_fuzziness_embedding)
                    arg_fuzziness_embedding = convert_to_axis(arg_fuzziness_embedding)

                    axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding,
                                                                   axis_r_embedding + axis_entity2_embedding + axis_time_embedding + axis_fuzziness_embedding,
                                                                   arg_r_embedding + arg_entity2_embedding + arg_time_embedding + arg_fuzziness_embedding)
                if query_structure[-1][i] == 'n':
                    idx += 1
                else:
                    idx += 4
        else:
            axis_embedding_list = []
            arg_embedding_list = []
            for i in range(len(query_structure)):
                axis_embedding, arg_embedding, idx = self.embed_query_cone_location(queries, query_structure[i], idx)
                axis_embedding_list.append(axis_embedding)
                arg_embedding_list.append(arg_embedding)

            stacked_axis_embeddings = torch.stack(axis_embedding_list)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)

            axis_embedding, arg_embedding = self.cone_intersection(stacked_axis_embeddings, stacked_arg_embeddings)

        return axis_embedding, arg_embedding, idx

    def embed_query_cone_fuzziness(self, queries, query_structure, idx):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in [('r', 'e', 't', 'l'), 'n', 'a', 'b']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(axis_entity_embedding)
                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding)
                idx += 1

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding
            else:
                axis_embedding, arg_embedding, idx = self.embed_query_cone_fuzziness(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    axis_embedding, arg_embedding = self.cone_negation(axis_embedding, arg_embedding)
                elif query_structure[-1][i] == 'a':
                    assert (queries[:, idx] == -3).all()
                    axis_embedding, arg_embedding = self.cone_after(axis_embedding, arg_embedding)
                elif query_structure[-1][i] == 'b':
                    assert (queries[:, idx] == -4).all()
                    axis_embedding, arg_embedding = self.cone_before(axis_embedding, arg_embedding)

                else:
                    axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, idx])
                    arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, idx])
                    axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                    arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)
                    axis_r_embedding = convert_to_axis(axis_r_embedding)
                    arg_r_embedding = convert_to_axis(arg_r_embedding)

                    axis_entity2_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx+1])
                    axis_entity2_embedding = self.angle_scale(axis_entity2_embedding, self.axis_scale)
                    axis_entity2_embedding = convert_to_axis(axis_entity2_embedding)
                    if self.use_cuda:
                        arg_entity2_embedding = torch.zeros_like(axis_entity2_embedding).cuda()
                    else:
                        arg_entity2_embedding = torch.zeros_like(axis_entity2_embedding)

                    axis_time_embedding = torch.index_select(self.axis_time_embedding, dim=0, index=queries[:, idx+2])
                    arg_time_embedding = torch.index_select(self.arg_time_embedding, dim=0, index=queries[:, idx+2])
                    axis_time_embedding = self.angle_scale(axis_time_embedding, self.axis_scale)
                    arg_time_embedding = self.angle_scale(arg_time_embedding, self.arg_scale)
                    axis_time_embedding = convert_to_axis(axis_time_embedding)
                    arg_time_embedding = convert_to_axis(arg_time_embedding)

                    axis_location_embedding = torch.index_select(self.axis_location_embedding, dim=0, index=queries[:, idx+3])
                    arg_location_embedding = torch.index_select(self.arg_location_embedding, dim=0, index=queries[:, idx+3])
                    axis_location_embedding = self.angle_scale(axis_location_embedding, self.axis_scale)
                    arg_location_embedding = self.angle_scale(arg_location_embedding, self.arg_scale)
                    axis_location_embedding = convert_to_axis(axis_location_embedding)
                    arg_location_embedding = convert_to_axis(arg_location_embedding)

                    axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding,
                                                                   axis_r_embedding + axis_entity2_embedding + axis_time_embedding + axis_location_embedding,
                                                                   arg_r_embedding + arg_entity2_embedding + arg_time_embedding + arg_location_embedding)
                if query_structure[-1][i] == 'n' or query_structure[-1][i] == 'a' or query_structure[-1][i] == 'b':
                    idx += 1
                else:
                    idx += 4
        else:
            axis_embedding_list = []
            arg_embedding_list = []
            for i in range(len(query_structure)):
                axis_embedding, arg_embedding, idx = self.embed_query_cone_fuzziness(queries, query_structure[i], idx)
                axis_embedding_list.append(axis_embedding)
                arg_embedding_list.append(arg_embedding)

            stacked_axis_embeddings = torch.stack(axis_embedding_list)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)

            axis_embedding, arg_embedding = self.cone_intersection(stacked_axis_embeddings, stacked_arg_embeddings)

        return axis_embedding, arg_embedding, idx

    def embed_query_cone(self, queries, query_structure, idx):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in [('r', 't', 'l', 'f'), 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(axis_entity_embedding)
                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding)
                idx += 1

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding
            else:
                axis_embedding, arg_embedding, idx = self.embed_query_cone(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    axis_embedding, arg_embedding = self.cone_negation(axis_embedding, arg_embedding)

                else:
                    axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, idx])
                    arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, idx])
                    axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                    arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)
                    axis_r_embedding = convert_to_axis(axis_r_embedding)
                    arg_r_embedding = convert_to_axis(arg_r_embedding)

                    axis_time_embedding = torch.index_select(self.axis_time_embedding, dim=0, index=queries[:, idx+1])
                    arg_time_embedding = torch.index_select(self.arg_time_embedding, dim=0, index=queries[:, idx+1])
                    axis_time_embedding = self.angle_scale(axis_time_embedding, self.axis_scale)
                    arg_time_embedding = self.angle_scale(arg_time_embedding, self.arg_scale)
                    axis_time_embedding = convert_to_axis(axis_time_embedding)
                    arg_time_embedding = convert_to_axis(arg_time_embedding)

                    axis_location_embedding = torch.index_select(self.axis_location_embedding, dim=0, index=queries[:, idx+2])
                    arg_location_embedding = torch.index_select(self.arg_location_embedding, dim=0, index=queries[:, idx+2])
                    axis_location_embedding = self.angle_scale(axis_location_embedding, self.axis_scale)
                    arg_location_embedding = self.angle_scale(arg_location_embedding, self.arg_scale)
                    axis_location_embedding = convert_to_axis(axis_location_embedding)
                    arg_location_embedding = convert_to_axis(arg_location_embedding)

                    axis_fuzziness_embedding = torch.index_select(self.axis_fuzziness_embedding, dim=0, index=queries[:, idx+3])
                    arg_fuzziness_embedding = torch.index_select(self.arg_fuzziness_embedding, dim=0, index=queries[:, idx+3])
                    axis_fuzziness_embedding = self.angle_scale(axis_fuzziness_embedding, self.axis_scale)
                    arg_fuzziness_embedding = self.angle_scale(arg_fuzziness_embedding, self.arg_scale)
                    axis_fuzziness_embedding = convert_to_axis(axis_fuzziness_embedding)
                    arg_fuzziness_embedding = convert_to_axis(arg_fuzziness_embedding)

                    axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding,
                                                                   axis_r_embedding + axis_time_embedding + axis_location_embedding + axis_fuzziness_embedding,
                                                                   arg_r_embedding + arg_time_embedding + arg_location_embedding + arg_fuzziness_embedding)
                if query_structure[-1][i] == 'n':
                    idx += 1
                else:
                    idx += 4
        else:
            axis_embedding_list = []
            arg_embedding_list = []
            for i in range(len(query_structure)):
                axis_embedding, arg_embedding, idx = self.embed_query_cone(queries, query_structure[i], idx)
                axis_embedding_list.append(axis_embedding)
                arg_embedding_list.append(arg_embedding)

            stacked_axis_embeddings = torch.stack(axis_embedding_list)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)

            axis_embedding, arg_embedding = self.cone_intersection(stacked_axis_embeddings, stacked_arg_embeddings)

        return axis_embedding, arg_embedding, idx

    def cal_logit_cone(self, entity_embedding, query_axis_embedding, query_arg_embedding):
        delta1 = entity_embedding - (query_axis_embedding - query_arg_embedding)
        delta2 = entity_embedding - (query_axis_embedding + query_arg_embedding)

        distance2axis = torch.abs(torch.sin((entity_embedding - query_axis_embedding) / 2))
        distance_base = torch.abs(torch.sin(query_arg_embedding / 2))

        indicator_in = distance2axis < distance_base
        distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
        distance_out[indicator_in] = 0.

        distance_in = torch.min(distance2axis, distance_base)

        distance = torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1)
        logit = self.gamma - distance * self.modulus

        return logit

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_axis_embeddings, all_arg_embeddings = [], [], []
        all_union_idxs, all_union_axis_embeddings, all_union_arg_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                if query_structure in [(('e', (('r', 'e', 'l', 'f'),)), ('e', (('r', 'e', 'l', 'f'),)), ('u',)),
                                       ]:
                    axis_embedding, arg_embedding, _ = \
                        self.embed_query_cone_time(
                            self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                            self.transform_union_structure(query_structure), 0)
                elif query_structure in [(('e', (('r', 'e', 't', 'f'),)), ('e', (('r', 'e', 't', 'f'),)), ('u',)),
                                         ]:
                    axis_embedding, arg_embedding, _ = \
                        self.embed_query_cone_location(
                            self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                            self.transform_union_structure(query_structure), 0)
                elif query_structure in [(('e', (('r', 'e', 't', 'l'),)), ('e', (('r', 'e', 't', 'l'),)), ('u',)),
                                         ]:
                    axis_embedding, arg_embedding, _ = \
                        self.embed_query_cone_fuzziness(
                            self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                            self.transform_union_structure(query_structure), 0)
                else:
                    axis_embedding, arg_embedding, _ = \
                        self.embed_query_cone(
                            self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                            self.transform_union_structure(query_structure), 0)

                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_axis_embeddings.append(axis_embedding)
                all_union_arg_embeddings.append(arg_embedding)
            else:
                if query_structure in [('e', (('r', 'e', 'l', 'f'),)),
                                       (('e', (('r', 'e', 'l', 'f'),)), ('e', (('r', 'e', 'l', 'f'), 'n'))),
                                       ('e', (('r', 'e', 'l', 'f'), 'a')),
                                       ('e', (('r', 'e', 'l', 'f'), 'b')),
                                       (('e', (('r', 'e', 'l', 'f'), 'a')), ('e', (('r', 'e', 'l', 'f'), 'b')))
                                       ]:
                    axis_embedding, arg_embedding, _ = self.embed_query_cone_time(batch_queries_dict[query_structure], query_structure, 0)
                elif query_structure in [('e', (('r', 'e', 't', 'f'),)),
                                         (('e', (('r', 'e', 't', 'f'),)), ('e', (('r', 'e', 't', 'f'), 'n')))]:
                    axis_embedding, arg_embedding, _ = self.embed_query_cone_location(batch_queries_dict[query_structure], query_structure, 0)
                elif query_structure in [('e', (('r', 'e', 't', 'l'),)),
                                         (('e', (('r', 'e', 't', 'l'),)), ('e', (('r', 'e', 't', 'l'), 'n'))),
                                         ('e', (('r', 'e', 't', 'l'), 'a')),
                                         ('e', (('r', 'e', 't', 'l'), 'b')),
                                         (('e', (('r', 'e', 't', 'l'), 'a')), ('e', (('r', 'e', 't', 'l'), 'b')))
                                         ]:
                    axis_embedding, arg_embedding, _ = self.embed_query_cone_fuzziness(batch_queries_dict[query_structure], query_structure, 0)
                else:
                    axis_embedding, arg_embedding, _ = self.embed_query_cone(batch_queries_dict[query_structure], query_structure, 0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_axis_embeddings.append(axis_embedding)
                all_arg_embeddings.append(arg_embedding)

        if len(all_axis_embeddings) > 0:
            all_axis_embeddings = torch.cat(all_axis_embeddings, dim=0).unsqueeze(1)
            all_arg_embeddings = torch.cat(all_arg_embeddings, dim=0).unsqueeze(1)
        if len(all_union_axis_embeddings) > 0:
            all_union_axis_embeddings = torch.cat(all_union_axis_embeddings, dim=0).unsqueeze(1)
            all_union_arg_embeddings = torch.cat(all_union_arg_embeddings, dim=0).unsqueeze(1)
            all_union_axis_embeddings = all_union_axis_embeddings.view(
                all_union_axis_embeddings.shape[0] // 2, 2, 1, -1)
            all_union_arg_embeddings = all_union_arg_embeddings.view(
                all_union_arg_embeddings.shape[0] // 2, 2, 1, -1)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)

                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_logit = self.cal_logit_cone(positive_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)


            if len(all_union_axis_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)

                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_union_logit = self.cal_logit_cone(positive_embedding, all_union_axis_embeddings, all_union_arg_embeddings)

                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1).long()).view(batch_size, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                negative_logit = self.cal_logit_cone(negative_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_axis_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1).long()).view(batch_size, 1, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                negative_union_logit = self.cal_logit_cone(negative_embedding, all_union_axis_embeddings, all_union_arg_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(ConeProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.time_dim = dim
        self.location_dim = dim
        self.fuzziness_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, source_embedding_axis, source_embedding_arg, r_embedding_axis, r_embedding_arg):
        x = torch.cat([source_embedding_axis + r_embedding_axis, source_embedding_arg + r_embedding_arg], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        axis, arg = torch.chunk(x, 2, dim=-1)
        axis_embeddings = convert_to_axis(axis)
        arg_embeddings = convert_to_arg(arg)
        return axis_embeddings, arg_embeddings


class ConeIntersection(nn.Module):
    def __init__(self, dim, drop):
        super(ConeIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, axis_embeddings, arg_embeddings):
        all_embeddings = torch.cat(
            [axis_embeddings, arg_embeddings], dim=-1)
        layer1_act = F.silu(self.layer1(all_embeddings))
        attention = F.softmax(self.layer2(layer1_act), dim=0)

        axis_embeddings = torch.sum(attention * axis_embeddings, dim=0)
        arg_embeddings = torch.sum(attention * arg_embeddings, dim=0)

        return axis_embeddings, arg_embeddings


class ConeNegation(nn.Module):
    def __init__(self):
        super(ConeNegation, self).__init__()

    def forward(self, axis_embedding, arg_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0

        axis_embedding[indicator_positive] = axis_embedding[indicator_positive] - pi
        axis_embedding[indicator_negative] = axis_embedding[indicator_negative] + pi

        arg_embedding = pi - arg_embedding

        return axis_embedding, arg_embedding


class ConeAfter(nn.Module):
    def __init__(self):
        super(ConeAfter, self).__init__()

    def forward(self, axis_embedding, arg_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0

        arg_embedding0 = arg_embedding
        arg_embedding[indicator_positive] = (pi - (arg_embedding0[indicator_positive] / 2 + axis_embedding[indicator_positive]))
        arg_embedding[indicator_negative] = (-arg_embedding0[indicator_negative] / 2 - axis_embedding[indicator_negative])

        axis_embedding[indicator_positive] = 2 * axis_embedding[indicator_positive] - (pi + arg_embedding0[indicator_positive] / 2)
        axis_embedding[indicator_negative] = axis_embedding[indicator_negative] / 2 + arg_embedding0[indicator_negative] / 4

        return axis_embedding, arg_embedding


class ConeBefore(nn.Module):
    def __init__(self):
        super(ConeBefore, self).__init__()

    def forward(self, axis_embedding, arg_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0

        arg_embedding0 = arg_embedding
        arg_embedding[indicator_positive] = (-arg_embedding0[indicator_positive] / 2 + axis_embedding[indicator_positive])
        arg_embedding[indicator_negative] = (pi - (arg_embedding0[indicator_negative] / 2 - axis_embedding[indicator_negative]))

        axis_embedding[indicator_positive] = axis_embedding[indicator_positive] / 2 - arg_embedding0[indicator_positive] / 4
        axis_embedding[indicator_negative] = 2 * axis_embedding[indicator_negative] + (pi + arg_embedding0[indicator_negative] / 2)

        return axis_embedding, arg_embedding