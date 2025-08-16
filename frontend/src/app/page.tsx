"use client";
import { useState, useRef } from "react";
import {
  Box, Grid, VStack, HStack, Text, Button,
  Textarea, Badge, Heading, Wrap, WrapItem,
  Card, Spinner, Separator
} from "@chakra-ui/react";

export default function Home() {
  const [targetText, setTargetText] = useState("Klicke auf Next Sentence, um zu starten.");
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [score, setScore] = useState<number | null>(null);
  const [mistakes, setMistakes] = useState<string[]>([]);
  const [tip, setTip] = useState("");
  const [transcript, setTranscript] = useState("");

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const handleRecord = async () => {
    if (!recording) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      mediaRecorder.start();
      setRecording(true);
    } else {
      mediaRecorderRef.current?.stop();
      mediaRecorderRef.current!.onstop = async () => {
        setRecording(false);
        setLoading(true);

        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");
        formData.append("target_text", targetText);

        try {
          const res = await fetch("/api/evaluate", { method: "POST", body: formData });
          const data = await res.json();
          setScore(data.score ?? null);
          setMistakes(Array.isArray(data.mistakes) ? data.mistakes : []);
          setTranscript(data.transcript ?? "");
          setTip(data.tip ?? "");
        } catch (err) {
          console.error("API error", err);
        } finally {
          setLoading(false);
        }
      };
    }
  };

  const nextSentence = async () => {
    setScore(null);
    setMistakes([]);
    setTranscript("");
    setTip("");

    try {
      const res = await fetch("/api/sentence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}), // không cần gì thêm, API đã fix prompt
      });
      const data = await res.json();
      setTargetText(data.sentence || "Keine Antwort");
    } catch (err) {
      console.error("Error fetching sentence", err);
      setTargetText("Fehler beim Abrufen des Satzes.");
    }
  };

  return (
    <Box minH="100vh" p={8} bg="gray.50" display="flex" justifyContent="center">
      <Grid templateColumns={{ base: "1fr", md: "repeat(2, 1fr)" }} gap={8} maxW="6xl" w="100%">
        {/* Recording Card */}
        <Card.Root p={6} rounded="2xl" shadow="sm" borderWidth="1px" bg="white">
          <Card.Header>
            <Heading size="md">🎤 Pronunciation Practice</Heading>
          </Card.Header>
          <Card.Body>
            <VStack align="start" gap={5}>
              <Text fontSize="xl" fontWeight="semibold">{targetText}</Text>
              <Button onClick={nextSentence} colorScheme="teal" size="sm">Next Sentence</Button>
              <HStack gap={4}>
                <Button
                  onClick={handleRecord}
                  colorScheme={recording ? "red" : "blue"}
                  rounded="full"
                  w="56px" h="56px"
                >
                  {recording ? "⏹" : "🎙"}
                </Button>
                <Text fontSize="sm">{recording ? "Recording…" : "Tap to start recording"}</Text>
              </HStack>
              <Textarea value={transcript} readOnly rows={3} placeholder="Transcript will appear here..." />
            </VStack>
          </Card.Body>
        </Card.Root>

        {/* Score & Feedback Card */}
        <Card.Root p={6} rounded="2xl" shadow="sm" borderWidth="1px" bg="white">
          <Card.Header>
            <Heading size="md">📊 Score & Feedback</Heading>
          </Card.Header>
          <Card.Body>
            {loading ? (
              <HStack justify="center" gap={4}>
                <Spinner size="lg" color="blue.400" />
                <Text>Analyzing your speech...</Text>
              </HStack>
            ) : score !== null ? (
              <VStack align="start" gap={5}>
                <Text fontSize="4xl" fontWeight="bold" color={score >= 80 ? "green.500" : "orange.400"}>{score}%</Text>
                <Box w="full">
                  <Text fontWeight="medium">Mistake words</Text>
                  <Wrap mt={2} gap={2}>
                    {mistakes.length > 0 ? mistakes.map((w, i) => (
                      <WrapItem key={i}><Badge colorScheme="red">{w}</Badge></WrapItem>
                    )) : <Text color="gray.400">No mistakes 🎉</Text>}
                  </Wrap>
                </Box>
                <Separator />
                <Box w="full">
                  <Text fontWeight="medium">Tip</Text>
                  <Text>{tip}</Text>
                </Box>
              </VStack>
            ) : (
              <Text color="gray.400">No score yet. Record and analyze.</Text>
            )}
          </Card.Body>
        </Card.Root>
      </Grid>
    </Box>
  );
}