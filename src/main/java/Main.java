import java.io.*;
import java.util.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import edu.stanford.nlp.io.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.ie.crf.*;
import edu.stanford.nlp.sequences.*;
import edu.stanford.nlp.ling.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.IOException;

import edu.stanford.nlp.classify.Classifier;
import edu.stanford.nlp.classify.ColumnDataClassifier;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.objectbank.ObjectBank;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.Pair;

import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.time.*;

public class Main {
    public static void main(String[] args) throws IOException, ClassNotFoundException{

        String[] text = {"", "h.to"};

        //trainAndWrite("model.txt", "test.properties", "testData.txt");
        CRFClassifier model = getModel("./model.txt");
        doTagging(model, text[1]);

        System.out.println();
        System.out.println("Training ColumnDataClassifier");
        ColumnDataClassifier cdc = new ColumnDataClassifier("/home/bill/Desktop/testCoreNlp/classify.properties");
        cdc.trainClassifier("/home/bill/Desktop/testCoreNlp/classify.txt");
        Datum<String,String> d = cdc.makeDatumFromStrings(text);
        System.out.printf("%s  ==>  %s (%.4f)%n", text[1], cdc.classOf(d), cdc.scoresOf(d).getCount(cdc.classOf(d)));

        // for (String line : ObjectBank.getLineIterator("/home/bill/Desktop/testCoreNlp/classify.test", "utf-8")) {
        //     Datum<String,String> d = cdc.makeDatumFromLine(line);
        //     System.out.printf("%s  ==>  %s (%.4f)%n", line, cdc.classOf(d), cdc.scoresOf(d).getCount(cdc.classOf(d)));
        // }

        // System.out.println();
        // System.out.println("Testing accuracy of ColumnDataClassifier");
        // Pair<Double, Double> performance = cdc.testClassifier("/home/bill/Desktop/testCoreNlp/classify.test");
        // System.out.printf("Accuracy: %.3f; macro-F1: %.3f%n", performance.first(), performance.second());
        System.out.println();

        DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        Date date = new Date();
        System.out.println(dateFormat.format(date));

        // Properties props = new Properties();
        // props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
        // props.setProperty("coref.algorithm", "neural");
        // props.setProperty("sutime.mark-TimeRanges", "true");
        // props.setProperty("sutime.includeRange", "true");
        // StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        // Annotation document = new Annotation(text[1]);
        // document.set(CoreAnnotations.DocDateAnnotation.class, dateFormat.format(date));
        // pipeline.annotate(document);

        // List<CoreMap> sentences = document.get(SentencesAnnotation.class);

        // for(CoreMap sentence: sentences){
        //     for(CoreLabel token: sentence.get(TokensAnnotation.class)){
        //         System.out.print(token.get(NamedEntityTagAnnotation.class) + "/" + token.get(NormalizedNamedEntityTagAnnotation.class) + " ");
        //     }
        //     System.out.println();
        // }
        long time1 = System.currentTimeMillis();

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
        props.setProperty("coref.algorithm", "neural");
        props.setProperty("ner.model", "./model.txt");
        props.setProperty("sutime.mark-TimeRanges", "true");
        props.setProperty("sutime.includeRange", "true");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        pipeline.addAnnotator(new TokenizerAnnotator(false));
        pipeline.addAnnotator(new WordsToSentencesAnnotator(false));
        pipeline.addAnnotator(new POSTaggerAnnotator(false));
        pipeline.addAnnotator(new TimeAnnotator("sutime", props));
        
        Annotation annotation = new Annotation(text[1]);
        annotation.set(CoreAnnotations.DocDateAnnotation.class, dateFormat.format(date));
        pipeline.annotate(annotation);
        System.out.println(annotation.get(CoreAnnotations.TextAnnotation.class));
        List<CoreMap> timexAnnsAll = annotation.get(TimeAnnotations.TimexAnnotations.class);
        for (CoreMap cm : timexAnnsAll) {
            List<CoreLabel> tokens = cm.get(CoreAnnotations.TokensAnnotation.class);
            System.out.println(cm + " --> " + cm.get(TimeExpression.Annotation.class).getTemporal());
        }
        long time2 = System.currentTimeMillis();
        System.out.println(time2-time1);

    }
    public static void trainAndWrite(String modelOutPath, String prop, String trainingFilePath){
        Properties props = StringUtils.propFileToProperties(prop);
        props.setProperty("serializeTo", modelOutPath);

        //if input use that, else use from properties file.
        if (trainingFilePath != null) {
            props.setProperty("trainFile", trainingFilePath);
        }

        SeqClassifierFlags flags = new SeqClassifierFlags(props);
        CRFClassifier<CoreLabel> crf = new CRFClassifier<>(flags);
        crf.train();

        crf.serializeClassifier(modelOutPath);
    }
    public static CRFClassifier getModel(String modelPath) {
        return CRFClassifier.getClassifierNoExceptions(modelPath);
    }
    public static void doTagging(CRFClassifier model, String input) {
        input = input.trim();
        System.out.println(input + " => "  +  model.classifyToString(input, "inlineXML", true));
    }            


}