FROM python:3.7

ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0
ARG MODEL_NAME
ARG TOKENIZE_MODEL
ARG VECTORIZE_MODEL
ARG LR_MODEL

ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN python3 -m spacy download en

ADD setup.py /app/setup.py
ADD models /app/models

ADD src /app/src
ADD reddit_train.csv /app/reddit_train.csv

WORKDIR /app

RUN pip install -e .

RUN if [ "x$TOKENIZE_MODEL" = "x" ] ; then echo TOKENIZE_MODEL Argument not provided ; else echo $TOKENIZE_MODEL >> /app/models/tokenized_location.txt ; fi
RUN if [ "x$VECTORIZE_MODEL" = "x" ] ; then echo VECTORIZE_MODEL Argument not provided ; else echo $VECTORIZE_MODEL >> /app/models/tfidf_vectorizer_location.txt ; fi
RUN if [ "x$LR_MODEL" = "x" ] ; then echo LR_MODEL Argument not provided ; else echo $LR_MODEL >> /app/models/lr_model_location.txt ; fi


EXPOSE 5000

CMD exec seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE