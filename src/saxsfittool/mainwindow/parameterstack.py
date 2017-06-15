class ParameterStack:
    def __init__(self):
        self._stack = {}
        self._pointers = {}
        self._model = None

    def addModel(self, model):
        self._stack[model]=[]
        self._pointers[model]=0

    def model(self):
        return self._model

    def setModel(self, model):
        if model not in self._stack:
            self.addModel(model)
        self._model = model

    def truncate(self):
        self._stack[self._model]=self._stack[self._model][:self._pointers[self._model]+1]

    @staticmethod
    def _deep_copy_parameters(parameters):
        return [p.copy() for p in parameters]

    def push(self, parameters):
        self.truncate()
        same_as_last = False
        if self._stack[self._model]:
            assert len(self._stack[self._model][-1]) == len(parameters)
            for oldp, newp in zip(self._stack[self._model][-1], parameters):
                assert oldp['name']==newp['name']
                for k in oldp:
                    if oldp[k]!=newp[k]:
                        break
                else:
                    continue
                break
            else:
                same_as_last=True
        if not same_as_last:
            self._stack[self._model].append(self._deep_copy_parameters(parameters))
            self._pointers[self._model]+=1

    def clear(self):
        self._stack[self._model]=[]
        self._pointers[self._model]=0

    def get(self):
        # print('++++++++ STACK IS:')
        # for i,pars in enumerate(self._stack[self._model]):
        #     print('-------')
        #     if i==self._pointers[self._model]-1:
        #         beg='> '
        #     else:
        #         beg=''
        #     print('\n'.join([beg+p['name']+': '+str(p['value']) for p in pars]))
        if self._pointers[self._model]==0:
            raise ValueError('Empty stack')
        assert self._pointers[self._model] <= len(self._stack[self._model])
        return self._stack[self._model][self._pointers[self._model]-1]

    def next(self):
        if self._pointers[self._model] >= len(self._stack[self._model]):
            self._pointers[self._model] = len(self._stack[self._model])
            raise ValueError('End of stack')
        else:
            self._pointers[self._model] +=1
            return self.get()

    def prev(self):
        if self._pointers[self._model] <=1:
            self._pointers[self._model] = 1
            raise ValueError('Beginning of stack')
        else:
            self._pointers[self._model] -=1
            return self.get()

    def setPointer(self, pos:int):
        if pos <0 or pos>=len(self._stack[self._model]):
            raise ValueError('Requested position out of range')
        self._pointers[self._model] = pos+1

    def pointer(self):
        return self._pointers[self._model] - 1

    def __len__(self):
        return len(self._stack[self._model])